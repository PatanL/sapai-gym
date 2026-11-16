"""Preserve identity-based turn restrictions and their observed counts."""


def include_retired_identities(identity_map, identities):
    # Python object IDs are positive. Negative placeholders retain retired
    # identities and history cardinality without matching any current entity.
    missing = sorted(set(identities) - identity_map.keys())
    identity_map.update({identity: -(index + 1) for index, identity in enumerate(missing)})
    return identity_map


def remap_shop_identities(original, copied, identities):
    identity_map = {
        id(left.obj): id(right.obj)
        for left, right in zip(original.shop, copied.shop)
        if not left.empty and not right.empty
    }
    include_retired_identities(identity_map, identities)
    return {identity_map[identity] for identity in identities}


def remap_team_orders(original, copied, orders):
    identity_map = {
        id(left.pet): id(right.pet)
        for left, right in zip(original.team, copied.team)
        if not left.empty and not right.empty
    }
    include_retired_identities(
        identity_map,
        {identity for order in orders for identity in order if identity is not None},
    )
    return {
        tuple(None if identity is None else identity_map[identity] for identity in order)
        for order in orders
    }
