from typing import Any, TypeAlias
import warnings

from django.contrib.auth.models import Permission
from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import FieldDoesNotExist
from django.db import models
from django.db.models import Model, Q, QuerySet

from guardian.conf import settings as guardian_settings
from guardian.core import ObjectPermissionChecker
from guardian.ctypes import get_content_type, get_default_content_type
from guardian.exceptions import MixedContentTypeError, ObjectNotPersisted

_PermType: TypeAlias = Permission | str

_DEFAULT_CONTENT_TYPE_PATH = f"{get_default_content_type.__module__}.{get_default_content_type.__name__}"


def _validate_homogeneous_objects(objects: QuerySet | list) -> tuple[QuerySet | list, type[Model]]:
    """Ensures a bulk target contains objects of a single model.

    Lists are materialized and validated eagerly, so a heterotypic object that
    only appears while the list is being consumed is rejected before any
    permission row is written.
    """
    if isinstance(objects, QuerySet):
        return objects, objects.model

    objects = list(objects)
    if not objects:
        return objects, None  # type: ignore[return-value]

    model = type(objects[0])
    for instance in objects[1:]:
        if type(instance) is not model:
            raise MixedContentTypeError(
                "Bulk object permission operations require every object to share the same model, "
                f"but got {model._meta.label} and {type(instance)._meta.label}."
            )
    return objects, model


def _dedupe_objects(objects: list[Model]) -> list[Model]:
    """Removes repeated references to the same object instance.

    Distinct instances sharing one primary key are kept apart by
    ``_validate_unique_pks``, so plain list repetition stays safe to collapse.
    """
    unique_objects = []
    seen_ids = set()
    for instance in objects:
        instance_id = id(instance)
        if instance_id in seen_ids:
            continue
        seen_ids.add(instance_id)
        unique_objects.append(instance)
    return unique_objects


def _validate_unique_pks(objects: list[Model]) -> None:
    """Rejects distinct instances that resolve to the same database row.

    Two different in-memory instances with the same pk would produce duplicate
    permission rows or delete one another's permissions, so such a batch is
    rejected before any write takes place.
    """
    seen_pks = set()
    for instance in objects:
        if instance.pk in seen_pks:
            raise MixedContentTypeError(
                "Bulk object permission operations received distinct instances of "
                f"{type(instance)._meta.label} with the same primary key ({instance.pk!r})."
            )
        seen_pks.add(instance.pk)


def _is_using_default_content_type() -> bool:
    """Check if default content type function is being used.

    Returns True if GUARDIAN_GET_CONTENT_TYPE setting points to the default
    get_default_content_type function, False otherwise.

    The check is necessary to avoid a regression where setting content_object
    in get_or_create defaults causes Django's GenericForeignKey.__set__ to
    overwrite the content_type that was explicitly set by a custom
    GUARDIAN_GET_CONTENT_TYPE function.

    Note: This function is not cached to allow dynamic changes during testing
    (e.g., when using mock.patch).
    """
    return guardian_settings.GET_CONTENT_TYPE == _DEFAULT_CONTENT_TYPE_PATH


def _ensure_permission(perm: _PermType, ctype: ContentType) -> Permission:
    if isinstance(perm, str):
        perm = Permission.objects.get(content_type=ctype, codename=perm)

    return perm


def _get_perm_filter(perm: _PermType, model: Model | type[Model] | None = None, ctype: ContentType | None = None) -> Q:
    if isinstance(perm, Permission):
        return Q(permission=perm)

    assert ctype is not None or model is not None
    if ctype is None and model is not None:
        ctype = get_content_type(model)

    return Q(permission__codename=perm, permission__content_type=ctype)


class BaseObjectPermissionManager(models.Manager):
    @property
    def user_or_group_field(self) -> str:
        try:
            self.model._meta.get_field("user")
            return "user"
        except FieldDoesNotExist:
            return "group"

    def is_generic(self) -> bool:
        try:
            self.model._meta.get_field("object_pk")
            return True
        except FieldDoesNotExist:
            return False

    def assign_perm(self, perm: _PermType, user_or_group: Any, obj: Model) -> Any:
        """Assigns permission with given `perm` for an instance `obj` and `user`."""
        if getattr(obj, "pk", None) is None:
            raise ObjectNotPersisted("Object %s needs to be persisted first" % obj)
        ctype = get_content_type(obj)
        permission = _ensure_permission(perm, ctype)

        kwargs = {"permission": permission, self.user_or_group_field: user_or_group}
        if self.is_generic():
            kwargs["content_type"] = ctype
            kwargs["object_pk"] = obj.pk
            if _is_using_default_content_type():
                kwargs["defaults"] = {"content_object": obj}
        else:
            kwargs["content_object"] = obj
        obj_perm, _ = self.get_or_create(**kwargs)
        return obj_perm

    def bulk_assign_perm(
        self, perm: _PermType, user_or_group: Any, queryset: QuerySet | list, ignore_conflicts: bool = False
    ) -> list:
        """
        Bulk assigns permissions with given `perm` for an objects in `queryset` and
        `user_or_group`.
        """
        objects, model = _validate_homogeneous_objects(queryset)
        if isinstance(objects, list):
            if not objects:
                return []
            objects = _dedupe_objects(objects)
            _validate_unique_pks(objects)

        ctype = get_content_type(model)

        permission = _ensure_permission(perm, ctype)
        checker = ObjectPermissionChecker(user_or_group)
        checker.prefetch_perms(objects)

        assigned_perms = []
        for instance in objects:
            if not checker.has_perm(permission.codename, instance):
                kwargs = {"permission": permission, self.user_or_group_field: user_or_group}
                if self.is_generic():
                    kwargs["content_type"] = ctype
                    kwargs["object_pk"] = instance.pk
                else:
                    kwargs["content_object"] = instance
                assigned_perms.append(self.model(**kwargs))
        self.model.objects.bulk_create(assigned_perms, ignore_conflicts=ignore_conflicts)

        return assigned_perms

    def assign_perm_to_many(
        self, perm: _PermType, users_or_groups: Any, obj: Model, ignore_conflicts: bool = False
    ) -> list:
        """
        Bulk assigns given `perm` for the object `obj` to a set of users or a set of groups.
        """
        ctype = get_content_type(obj)
        permission = _ensure_permission(perm, ctype)

        kwargs = {"permission": permission}
        if self.is_generic():
            kwargs["content_type"] = ctype
            kwargs["object_pk"] = obj.pk
        else:
            kwargs["content_object"] = obj

        to_add = []
        field = self.user_or_group_field
        for user in users_or_groups:
            kwargs[field] = user
            to_add.append(self.model(**kwargs))

        return self.model.objects.bulk_create(to_add, ignore_conflicts=ignore_conflicts)

    def assign(self, perm: _PermType, user_or_group: Any, obj: Model) -> Any:
        """Depreciated function name left in for compatibility"""
        warnings.warn(
            "UserObjectPermissionManager method 'assign' is being renamed to 'assign_perm'. Update your code accordingly as old name will be depreciated in 2.0 version.",
            DeprecationWarning,
        )
        return self.assign_perm(perm, user_or_group, obj)

    def remove_perm(self, perm: _PermType, user_or_group: Any, obj: Model) -> tuple[int, dict]:
        """
        Removes permission `perm` for an instance `obj` and given `user_or_group`.

        Uses `QuerySet.delete()`, which sends `pre_delete` and `post_delete`
        signals but does not call each object-permission instance's `delete()` method.
        """
        if getattr(obj, "pk", None) is None:
            raise ObjectNotPersisted("Object %s needs to be persisted first" % obj)

        filters = Q(**{self.user_or_group_field: user_or_group})
        filters &= _get_perm_filter(perm, model=obj)
        if self.is_generic():
            filters &= Q(object_pk=obj.pk)
        else:
            filters &= Q(content_object__pk=obj.pk)
        return self.filter(filters).delete()

    def bulk_remove_perm(self, perm: _PermType, user_or_group: Any, queryset: QuerySet | list) -> tuple[int, dict]:
        """
        Removes permission `perm` for a `queryset` and given `user_or_group`.

        Uses `QuerySet.delete()`, which sends `pre_delete` and `post_delete`
        signals but does not call each object-permission instance's `delete()` method.
        """
        objects, model = _validate_homogeneous_objects(queryset)
        if isinstance(objects, list):
            if not objects:
                return (0, {})
            objects = _dedupe_objects(objects)
            _validate_unique_pks(objects)

        ctype = get_content_type(model)

        filters = Q(**{self.user_or_group_field: user_or_group})
        filters &= _get_perm_filter(perm, ctype=ctype)
        if self.is_generic():
            if isinstance(objects, list):
                filters &= Q(object_pk__in=[str(obj.pk) for obj in objects])
            else:
                filters &= Q(object_pk__in=[str(pk) for pk in objects.values_list("pk", flat=True)])
        else:
            filters &= Q(content_object__in=objects)

        return self.filter(filters).delete()

    def remove_perm_from_many(self, perm: _PermType, users_or_groups: Any, obj: Model) -> tuple[int, dict]:
        """
        Bulk removes given `perm` for the object `obj` from a set of users or a set of groups.
        """
        ctype = get_content_type(obj)
        filters = _get_perm_filter(perm, ctype=ctype)
        if self.is_generic():
            filters &= Q(object_pk=str(obj.pk))
        else:
            filters &= Q(content_object=obj)

        if isinstance(users_or_groups, list):
            to_remove = [item.pk for item in users_or_groups]
        else:
            to_remove = users_or_groups.values_list("pk", flat=True)

        filters &= Q(**{f"{self.user_or_group_field}_id__in": to_remove})

        return self.filter(filters).delete()


class UserObjectPermissionManager(BaseObjectPermissionManager):
    """
    See Also:
        `guardian.managers.UserObjectPermissionManager`
    """


class GroupObjectPermissionManager(BaseObjectPermissionManager):
    """
    See Also:
        `guardian.managers.UserObjectPermissionManager`
    """
