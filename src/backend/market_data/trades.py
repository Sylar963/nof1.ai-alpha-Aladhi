"""Price-forming public prints, with exact decimals and exchange timestamps."""

from decimal import Decimal, InvalidOperation


def trade_print(payload):
    if payload.get('trade_type') not in {'FILL', 'RPI'}:
        return None
    created = payload.get('created_at_us')
    if created is None and type(payload.get('created_at')) is int:
        created = payload['created_at'] * 1000
    if type(created) is not int or created <= 0:
        raise ValueError('trade is missing its matching-engine timestamp')
    try:
        price, size = Decimal(str(payload['price'])), Decimal(str(payload['size']))
    except (KeyError, InvalidOperation) as exc:
        raise ValueError('invalid trade price or quantity') from exc
    if not price.is_finite() or not size.is_finite() or price <= 0 or size <= 0:
        raise ValueError('trade price and quantity must be finite and positive')
    identifier = payload.get('trade_id_low64')
    if identifier is None:
        full_id = payload.get('id')
        if not isinstance(full_id, str) or not full_id.isdecimal():
            raise ValueError('trade is missing an identity')
        value = int(full_id) % (1 << 64)
        identifier = str(value if value < (1 << 63) else value - (1 << 64))
    identity = '|'.join((str(identifier), payload['trade_type'], str(created),
                         str(price.normalize()), str(size.normalize())))
    return created * 1000, price, size, identity
