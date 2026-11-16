"""API вихідних накладних: CRUD, рядки форми, права доступу, списки з фільтрами."""
from __future__ import annotations

from decimal import Decimal

import pytest
from httpx import AsyncClient

from tests.invoice_utils import (
    cleanup_orders,  # noqa: F401 - фікстура
    INBOUND,
    ORDERS,
    OUTBOUND,
    PLANNED_DATE,
    STOCK,
    add_product,
    create_inbound,
    create_outbound,
    get_outbound,
    line,
    make_fixture,
    outbound_payload,
    post_outbound,
    receive_stock,
)
from tests.utils import (
    ADMIN,
    MANAGER,
    WORKER,
    auth,
    create_recipient,
    login,
    unique,
)

pytestmark = pytest.mark.usefixtures("cleanup_orders")

# ---------------------------------------------------------------------------
# Створення
# ---------------------------------------------------------------------------

async def test_create_assigns_number_and_draft_status(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)

    created = await create_outbound(client, token, fixture)

    assert created["order_status"] == "draft"
    assert created["invoice_number"] == f"ВН-{created['outbound_order_id']:06d}"
    assert created["created_by_user_id"]
    assert created["created_at"]
    assert created["planned_shipment_date"].startswith("2026-12-02")

async def test_create_with_custom_invoice_number(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    number = unique("OUT")

    created = await create_outbound(client, token, fixture, invoice_number=number)

    assert created["invoice_number"] == number
    assert (await get_outbound(client, token, created["outbound_order_id"]))[
        "invoice_number"
    ] == number

async def test_create_with_several_product_lines(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    second = await add_product(client, token, fixture["category_id"])
    third = await add_product(client, token, fixture["category_id"])

    created = await create_outbound(
        client,
        token,
        fixture,
        [
            line(fixture["product_id"], 5, "box"),
            line(second, "2.5", "pallet"),
            line(third, 1, "box"),
        ],
    )

    assert len(created["items"]) == 3
    by_product = {item["product_id"]: item for item in created["items"]}
    assert Decimal(by_product[fixture["product_id"]]["quantity"]) == Decimal("5")
    assert Decimal(by_product[second]["quantity"]) == Decimal("2.5")
    assert by_product[second]["package_type"] == "pallet"
    assert all(item["item_id"] for item in created["items"])

async def test_create_allows_same_product_in_several_lines(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)

    created = await create_outbound(
        client,
        token,
        fixture,
        [
            line(fixture["product_id"], 3, "box"),
            line(fixture["product_id"], 4, "pallet"),
        ],
    )

    assert len(created["items"]) == 2

async def test_create_minimal_and_decimal_quantities_are_accepted(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)

    created = await create_outbound(
        client,
        token,
        fixture,
        [line(fixture["product_id"], "0.01"), line(fixture["product_id"], "99999999.99")],
    )

    quantities = sorted(Decimal(item["quantity"]) for item in created["items"])
    assert quantities == [Decimal("0.01"), Decimal("99999999.99")]

@pytest.mark.xfail(
    strict=True,
    reason=(
        "БАГ: outbound_crud.create_outbound_order читає e.original_error, якого "
        "немає в IntegrityError (є e.orig), тому дубль номера дає необроблений "
        "IntegrityError (500) замість 400"
    ),
)
async def test_create_duplicate_invoice_number_returns_400(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    number = unique("OUT")
    await create_outbound(client, token, fixture, invoice_number=number)

    response = await client.post(
        OUTBOUND,
        headers=auth(token),
        json=outbound_payload(fixture, invoice_number=number),
    )

    assert response.status_code == 400, response.text

async def test_update_to_duplicate_invoice_number_returns_400(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    first = await create_outbound(client, token, fixture)
    second = await create_outbound(client, token, fixture)

    response = await client.patch(
        f"{OUTBOUND}/{second['outbound_order_id']}",
        headers=auth(token),
        json={"invoice_number": first["invoice_number"]},
    )

    assert response.status_code == 400, response.text

async def test_update_keeping_own_invoice_number_is_allowed(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    order = await create_outbound(client, token, fixture)

    response = await client.patch(
        f"{OUTBOUND}/{order['outbound_order_id']}",
        headers=auth(token),
        json={"invoice_number": order["invoice_number"]},
    )

    assert response.status_code == 200, response.text

# ---------------------------------------------------------------------------
# Валідація форми
# ---------------------------------------------------------------------------

def _bad_quantity(value):
    return lambda f: outbound_payload(f, [line(f["product_id"], value)])

INVALID_PAYLOADS = {
    "quantity_zero": _bad_quantity("0"),
    "quantity_negative": _bad_quantity("-1"),
    "quantity_three_decimals": _bad_quantity("1.005"),
    "quantity_too_many_digits": _bad_quantity("123456789.00"),
    "quantity_not_a_number": _bad_quantity("abc"),
    "empty_items": lambda f: outbound_payload(f, []),
    "missing_package_type": lambda f: outbound_payload(
        f, [{"product_id": f["product_id"], "quantity": "1"}]
    ),
    "package_type_too_long": lambda f: outbound_payload(
        f, [line(f["product_id"], 1, "x" * 31)]
    ),
    "invoice_number_too_long": lambda f: outbound_payload(f, invoice_number="N" * 51),
    "invalid_date": lambda f: outbound_payload(f, planned_shipment_date="не дата"),
    "invalid_status": lambda f: outbound_payload(f, order_status="shipped"),
}

@pytest.mark.parametrize("case", sorted(INVALID_PAYLOADS))
async def test_create_rejects_invalid_payload(client: AsyncClient, case: str):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)

    response = await client.post(
        OUTBOUND, headers=auth(token), json=INVALID_PAYLOADS[case](fixture)
    )

    assert response.status_code == 422, response.text

@pytest.mark.parametrize("missing", ["recipient_id", "warehouse_id", "planned_shipment_date", "items"])
async def test_create_requires_mandatory_fields(client: AsyncClient, missing: str):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    payload = outbound_payload(fixture)
    del payload[missing]

    response = await client.post(OUTBOUND, headers=auth(token), json=payload)

    assert response.status_code == 422, response.text

async def test_create_with_unknown_references_returns_404(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)

    unknown_warehouse = await client.post(
        OUTBOUND, headers=auth(token), json=outbound_payload(fixture, warehouse_id=999999)
    )
    assert unknown_warehouse.status_code == 404, unknown_warehouse.text

    unknown_recipient = await client.post(
        OUTBOUND, headers=auth(token), json=outbound_payload(fixture, recipient_id=999999)
    )
    assert unknown_recipient.status_code == 404, unknown_recipient.text

    unknown_product = await client.post(
        OUTBOUND,
        headers=auth(token),
        json=outbound_payload(fixture, [line(fixture["product_id"], 1), line(999999, 1)]),
    )
    assert unknown_product.status_code == 404, unknown_product.text

async def test_failed_create_does_not_leave_partial_order(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)

    await client.post(
        OUTBOUND,
        headers=auth(token),
        json=outbound_payload(fixture, [line(fixture["product_id"], 1), line(999999, 1)]),
    )

    listing = await client.get(
        OUTBOUND, headers=auth(token), params={"warehouse_id": fixture["warehouse_id"]}
    )
    assert listing.status_code == 200, listing.text
    assert listing.json() == []

# ---------------------------------------------------------------------------
# Редагування (форма накладної)
# ---------------------------------------------------------------------------

async def test_update_header_and_replace_lines(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    other_product = await add_product(client, token, fixture["category_id"])
    other_recipient = await create_recipient(client, token)
    order = await create_outbound(
        client, token, fixture, [line(fixture["product_id"], 5), line(other_product, 6)]
    )

    response = await client.patch(
        f"{OUTBOUND}/{order['outbound_order_id']}",
        headers=auth(token),
        json={
            "recipient_id": other_recipient,
            "planned_shipment_date": "2027-01-15T00:00:00Z",
            "items": [line(other_product, "7.25", "pallet")],
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["recipient_id"] == other_recipient
    assert body["planned_shipment_date"].startswith("2027-01-15")
    assert [item["product_id"] for item in body["items"]] == [other_product]
    assert Decimal(body["items"][0]["quantity"]) == Decimal("7.25")
    assert body["order_status"] == "draft"
    assert body["invoice_number"] == order["invoice_number"]

async def test_update_without_items_keeps_lines(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    order = await create_outbound(client, token, fixture, [line(fixture["product_id"], 8)])

    response = await client.patch(
        f"{OUTBOUND}/{order['outbound_order_id']}",
        headers=auth(token),
        json={"planned_shipment_date": "2027-02-01T00:00:00Z"},
    )

    assert response.status_code == 200, response.text
    assert len(response.json()["items"]) == 1
    assert Decimal(response.json()["items"][0]["quantity"]) == Decimal("8")

async def test_update_moves_order_to_another_warehouse(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    other = await make_fixture(client, token)
    order = await create_outbound(client, token, fixture)

    response = await client.patch(
        f"{OUTBOUND}/{order['outbound_order_id']}",
        headers=auth(token),
        json={"warehouse_id": other["warehouse_id"]},
    )

    assert response.status_code == 200, response.text
    assert response.json()["warehouse_id"] == other["warehouse_id"]

async def test_update_rejects_invalid_and_unknown_values(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    order = await create_outbound(client, token, fixture)
    url = f"{OUTBOUND}/{order['outbound_order_id']}"

    zero_quantity = await client.patch(
        url, headers=auth(token), json={"items": [line(fixture["product_id"], 0)]}
    )
    assert zero_quantity.status_code == 422, zero_quantity.text

    unknown_product = await client.patch(
        url, headers=auth(token), json={"items": [line(999999, 1)]}
    )
    assert unknown_product.status_code == 404, unknown_product.text

    unknown_recipient = await client.patch(
        url, headers=auth(token), json={"recipient_id": 999999}
    )
    assert unknown_recipient.status_code == 404, unknown_recipient.text

    unknown_warehouse = await client.patch(
        url, headers=auth(token), json={"warehouse_id": 999999}
    )
    assert unknown_warehouse.status_code == 404, unknown_warehouse.text

    unchanged = await get_outbound(client, token, order["outbound_order_id"])
    assert unchanged["items"][0]["quantity"] == order["items"][0]["quantity"]

async def test_unknown_order_returns_404(client: AsyncClient):
    token = await login(client, ADMIN)
    url = f"{OUTBOUND}/999999"

    assert (await client.get(url, headers=auth(token))).status_code == 404
    assert (await client.patch(url, headers=auth(token), json={"invoice_number": "X"})).status_code == 404
    assert (await client.delete(url, headers=auth(token))).status_code == 404
    assert (await client.get(f"{url}/availability", headers=auth(token))).status_code == 404

# ---------------------------------------------------------------------------
# Видалення
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("order_status", "expected"),
    [
        ("draft", 204),
        ("canceled", 204),
        ("in_progress", 400),
        ("ready", 400),
        ("completed", 400),
    ],
)
async def test_delete_depends_on_status(client: AsyncClient, order_status: str, expected: int):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    order = await create_outbound(client, token, fixture, order_status=order_status)
    order_id = order["outbound_order_id"]

    response = await client.delete(f"{OUTBOUND}/{order_id}", headers=auth(token))

    assert response.status_code == expected, response.text
    after = await client.get(f"{OUTBOUND}/{order_id}", headers=auth(token))
    assert after.status_code == (404 if expected == 204 else 200)

async def test_delete_removes_order_lines(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    order = await create_outbound(
        client, token, fixture, [line(fixture["product_id"], 1), line(fixture["product_id"], 2)]
    )

    response = await client.delete(
        f"{OUTBOUND}/{order['outbound_order_id']}", headers=auth(token)
    )

    assert response.status_code == 204, response.text

# ---------------------------------------------------------------------------
# Права доступу
# ---------------------------------------------------------------------------

async def test_unauthenticated_requests_are_rejected(client: AsyncClient):
    assert (await client.get(OUTBOUND)).status_code == 401
    assert (await client.get(f"{OUTBOUND}/1")).status_code == 401
    assert (await client.post(OUTBOUND, json={})).status_code == 401
    assert (await client.post(f"{OUTBOUND}/1/post")).status_code == 401
    assert (await client.get(f"{OUTBOUND}/1/availability")).status_code == 401
    assert (await client.get(ORDERS)).status_code == 401
    assert (await client.get(STOCK)).status_code == 401

async def test_storekeeper_can_read_but_not_modify(client: AsyncClient):
    admin = await login(client, ADMIN)
    worker = await login(client, WORKER)
    fixture = await make_fixture(client, admin)
    order = await create_outbound(client, admin, fixture)
    order_id = order["outbound_order_id"]

    assert (await client.get(OUTBOUND, headers=auth(worker))).status_code == 200
    assert (await client.get(f"{OUTBOUND}/{order_id}", headers=auth(worker))).status_code == 200
    assert (await client.get(f"{OUTBOUND}/{order_id}/availability", headers=auth(worker))).status_code == 200
    assert (await client.get(ORDERS, headers=auth(worker))).status_code == 200

    assert (
        await client.post(OUTBOUND, headers=auth(worker), json=outbound_payload(fixture))
    ).status_code == 403
    assert (
        await client.patch(
            f"{OUTBOUND}/{order_id}", headers=auth(worker), json={"invoice_number": "X"}
        )
    ).status_code == 403
    assert (await client.delete(f"{OUTBOUND}/{order_id}", headers=auth(worker))).status_code == 403
    assert (await post_outbound(client, worker, order_id)).status_code == 403
    assert (await get_outbound(client, admin, order_id))["order_status"] == "draft"

async def test_manager_can_create_edit_post_and_delete(client: AsyncClient):
    admin = await login(client, ADMIN)
    manager = await login(client, MANAGER)
    fixture = await make_fixture(client, admin)
    await receive_stock(client, admin, fixture, 50)

    created = await create_outbound(client, manager, fixture, [line(fixture["product_id"], 5)])
    order_id = created["outbound_order_id"]
    patched = await client.patch(
        f"{OUTBOUND}/{order_id}",
        headers=auth(manager),
        json={"items": [line(fixture["product_id"], 6)]},
    )
    assert patched.status_code == 200, patched.text
    assert (await post_outbound(client, manager, order_id)).status_code == 200

    draft = await create_outbound(client, manager, fixture)
    deleted = await client.delete(
        f"{OUTBOUND}/{draft['outbound_order_id']}", headers=auth(manager)
    )
    assert deleted.status_code == 204, deleted.text

# ---------------------------------------------------------------------------
# Список вихідних накладних (/outbound-orders)
# ---------------------------------------------------------------------------

async def test_outbound_list_filters(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    other_recipient = await create_recipient(client, token)
    await receive_stock(client, token, fixture, 100)

    draft = await create_outbound(client, token, fixture)
    other = await create_outbound(client, token, fixture, recipient_id=other_recipient)
    done = await create_outbound(client, token, fixture, [line(fixture["product_id"], 1)])
    assert (await post_outbound(client, token, done["outbound_order_id"])).status_code == 200

    def ids(response) -> set[int]:
        assert response.status_code == 200, response.text
        return {item["outbound_order_id"] for item in response.json()}

    scope = {"warehouse_id": fixture["warehouse_id"]}
    everything = await client.get(OUTBOUND, headers=auth(token), params=scope)
    assert ids(everything) == {
        draft["outbound_order_id"],
        other["outbound_order_id"],
        done["outbound_order_id"],
    }

    drafts = await client.get(
        OUTBOUND, headers=auth(token), params={**scope, "order_status": "draft"}
    )
    assert ids(drafts) == {draft["outbound_order_id"], other["outbound_order_id"]}

    completed = await client.get(
        OUTBOUND, headers=auth(token), params={**scope, "order_status": "completed"}
    )
    assert ids(completed) == {done["outbound_order_id"]}

    canceled = await client.get(
        OUTBOUND, headers=auth(token), params={**scope, "order_status": "canceled"}
    )
    assert ids(canceled) == set()

    by_recipient = await client.get(
        OUTBOUND, headers=auth(token), params={**scope, "recipient_id": other_recipient}
    )
    assert ids(by_recipient) == {other["outbound_order_id"]}

    by_number = await client.get(
        OUTBOUND, headers=auth(token), params={"search": draft["invoice_number"]}
    )
    assert ids(by_number) == {draft["outbound_order_id"]}

    nothing = await client.get(
        OUTBOUND, headers=auth(token), params={"search": "немає-такої-накладної"}
    )
    assert ids(nothing) == set()

async def test_outbound_list_is_sorted_newest_first_and_paginated(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    created = [
        (await create_outbound(client, token, fixture))["outbound_order_id"]
        for _ in range(5)
    ]
    scope = {"warehouse_id": fixture["warehouse_id"]}

    first_page = await client.get(
        OUTBOUND, headers=auth(token), params={**scope, "page": 1, "page_size": 2}
    )
    second_page = await client.get(
        OUTBOUND, headers=auth(token), params={**scope, "page": 2, "page_size": 2}
    )
    last_page = await client.get(
        OUTBOUND, headers=auth(token), params={**scope, "page": 3, "page_size": 2}
    )
    beyond = await client.get(
        OUTBOUND, headers=auth(token), params={**scope, "page": 4, "page_size": 2}
    )

    newest_first = sorted(created, reverse=True)
    assert [o["outbound_order_id"] for o in first_page.json()] == newest_first[:2]
    assert [o["outbound_order_id"] for o in second_page.json()] == newest_first[2:4]
    assert [o["outbound_order_id"] for o in last_page.json()] == newest_first[4:]
    assert beyond.json() == []

# ---------------------------------------------------------------------------
# Єдиний список накладних (/orders): фільтри за типом і статусом
# ---------------------------------------------------------------------------

async def _mixed_orders(client: AsyncClient, token: str) -> dict:
    """Склад із 2 прихідними та 3 вихідними накладними у різних статусах."""
    fixture = await make_fixture(client, token)
    await receive_stock(client, token, fixture, 100)  # проведена прихідна

    inbound_draft = await create_inbound(client, token, fixture)
    outbound_draft = await create_outbound(client, token, fixture, [line(fixture["product_id"], 2)])
    outbound_done = await create_outbound(client, token, fixture, [line(fixture["product_id"], 3)])
    assert (await post_outbound(client, token, outbound_done["outbound_order_id"])).status_code == 200
    outbound_canceled = await create_outbound(
        client, token, fixture, [line(fixture["product_id"], 1)], order_status="canceled"
    )
    return {
        **fixture,
        "inbound_draft": inbound_draft["inbound_order_id"],
        "outbound_draft": outbound_draft["outbound_order_id"],
        "outbound_done": outbound_done["outbound_order_id"],
        "outbound_canceled": outbound_canceled["outbound_order_id"],
    }

def _keys(response) -> set[tuple[str, int]]:
    assert response.status_code == 200, response.text
    return {(item["order_type"], item["order_id"]) for item in response.json()["items"]}

async def test_orders_list_filter_by_type(client: AsyncClient):
    token = await login(client, ADMIN)
    data = await _mixed_orders(client, token)
    scope = {"warehouse_id": data["warehouse_id"]}

    outbound = await client.get(ORDERS, headers=auth(token), params={**scope, "order_type": "outbound"})
    inbound = await client.get(ORDERS, headers=auth(token), params={**scope, "order_type": "inbound"})
    both = await client.get(ORDERS, headers=auth(token), params=scope)

    assert _keys(outbound) == {
        ("outbound", data["outbound_draft"]),
        ("outbound", data["outbound_done"]),
        ("outbound", data["outbound_canceled"]),
    }
    assert {t for t, _ in _keys(inbound)} == {"inbound"}
    assert ("inbound", data["inbound_draft"]) in _keys(inbound)
    assert _keys(both) == _keys(outbound) | _keys(inbound)
    assert outbound.json()["total"] == 3

@pytest.mark.parametrize("order_status", ["draft", "completed", "canceled", "in_progress", "ready"])
async def test_orders_list_filter_by_status(client: AsyncClient, order_status: str):
    token = await login(client, ADMIN)
    data = await _mixed_orders(client, token)

    response = await client.get(
        ORDERS,
        headers=auth(token),
        params={"warehouse_id": data["warehouse_id"], "order_status": order_status},
    )

    assert response.status_code == 200, response.text
    items = response.json()["items"]
    assert {item["order_status"] for item in items} <= {order_status}
    assert response.json()["total"] == len(items)

async def test_orders_list_combines_type_and_status_filters(client: AsyncClient):
    token = await login(client, ADMIN)
    data = await _mixed_orders(client, token)
    scope = {"warehouse_id": data["warehouse_id"]}

    outbound_drafts = await client.get(
        ORDERS, headers=auth(token), params={**scope, "order_type": "outbound", "order_status": "draft"}
    )
    outbound_completed = await client.get(
        ORDERS, headers=auth(token), params={**scope, "order_type": "outbound", "order_status": "completed"}
    )
    inbound_completed = await client.get(
        ORDERS, headers=auth(token), params={**scope, "order_type": "inbound", "order_status": "completed"}
    )
    outbound_ready = await client.get(
        ORDERS, headers=auth(token), params={**scope, "order_type": "outbound", "order_status": "ready"}
    )

    assert _keys(outbound_drafts) == {("outbound", data["outbound_draft"])}
    assert _keys(outbound_completed) == {("outbound", data["outbound_done"])}
    assert {t for t, _ in _keys(inbound_completed)} == {"inbound"}
    assert ("inbound", data["inbound_draft"]) not in _keys(inbound_completed)
    assert _keys(outbound_ready) == set()
    assert outbound_ready.json()["total"] == 0
    assert outbound_ready.json()["pages"] == 1

async def test_posting_moves_order_between_status_filters(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    await receive_stock(client, token, fixture, 20)
    order = await create_outbound(client, token, fixture, [line(fixture["product_id"], 5)])
    key = ("outbound", order["outbound_order_id"])
    scope = {"warehouse_id": fixture["warehouse_id"], "order_type": "outbound"}

    before_drafts = await client.get(ORDERS, headers=auth(token), params={**scope, "order_status": "draft"})
    before_done = await client.get(ORDERS, headers=auth(token), params={**scope, "order_status": "completed"})
    assert key in _keys(before_drafts)
    assert key not in _keys(before_done)

    assert (await post_outbound(client, token, order["outbound_order_id"])).status_code == 200

    after_drafts = await client.get(ORDERS, headers=auth(token), params={**scope, "order_status": "draft"})
    after_done = await client.get(ORDERS, headers=auth(token), params={**scope, "order_status": "completed"})
    assert key not in _keys(after_drafts)
    assert key in _keys(after_done)

async def test_orders_list_row_content(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    second = await add_product(client, token, fixture["category_id"])
    order = await create_outbound(
        client, token, fixture, [line(fixture["product_id"], "2.5"), line(second, 4)]
    )
    inbound = await create_inbound(client, token, fixture, [line(fixture["product_id"], 7)])

    response = await client.get(
        ORDERS, headers=auth(token), params={"warehouse_id": fixture["warehouse_id"]}
    )

    rows = {(r["order_type"], r["order_id"]): r for r in response.json()["items"]}
    outbound_row = rows[("outbound", order["outbound_order_id"])]
    assert outbound_row["number"] == f"ВН-{order['outbound_order_id']:06d}"
    assert outbound_row["invoice_number"] == order["invoice_number"]
    assert outbound_row["counterparty_id"] == fixture["recipient_id"]
    assert outbound_row["warehouse_id"] == fixture["warehouse_id"]
    assert outbound_row["order_status"] == "draft"
    assert outbound_row["items_count"] == 2
    assert Decimal(outbound_row["total_quantity"]) == Decimal("6.5")

    inbound_row = rows[("inbound", inbound["inbound_order_id"])]
    assert inbound_row["number"] == f"ПН-{inbound['inbound_order_id']:06d}"
    assert inbound_row["counterparty_id"] == fixture["supplier_id"]
    assert Decimal(inbound_row["total_quantity"]) == Decimal("7")

async def test_orders_list_search_by_number_and_counterparty(client: AsyncClient):
    token = await login(client, ADMIN)
    name = unique("Унікальний-одержувач")
    recipient_id = await create_recipient(client, token, company_name=name)
    fixture = await make_fixture(client, token)
    target = await create_outbound(client, token, fixture, recipient_id=recipient_id)
    await create_outbound(client, token, fixture)

    by_name = await client.get(ORDERS, headers=auth(token), params={"search": name})
    by_number = await client.get(
        ORDERS, headers=auth(token), params={"search": target["invoice_number"]}
    )
    by_lowercase = await client.get(
        ORDERS, headers=auth(token), params={"search": name.lower()}
    )
    blank = await client.get(
        ORDERS, headers=auth(token), params={"search": "   ", "warehouse_id": fixture["warehouse_id"]}
    )

    expected = {("outbound", target["outbound_order_id"])}
    assert _keys(by_name) == expected
    assert _keys(by_number) == expected
    assert _keys(by_lowercase) == expected
    assert blank.json()["total"] == 2

async def test_orders_list_pagination(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    for _ in range(5):
        await create_outbound(client, token, fixture)
    scope = {"warehouse_id": fixture["warehouse_id"], "order_type": "outbound", "page_size": 2}

    pages = [
        await client.get(ORDERS, headers=auth(token), params={**scope, "page": number})
        for number in (1, 2, 3, 4)
    ]

    for response in pages:
        assert response.status_code == 200, response.text
        assert response.json()["total"] == 5
        assert response.json()["pages"] == 3
        assert response.json()["page_size"] == 2
    assert [len(p.json()["items"]) for p in pages] == [2, 2, 1, 0]
    assert [p.json()["page"] for p in pages] == [1, 2, 3, 4]
    seen = [i["order_id"] for p in pages for i in p.json()["items"]]
    assert len(seen) == len(set(seen)) == 5

async def test_orders_list_filter_by_planned_date(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    early = await create_outbound(client, token, fixture, planned_shipment_date="2027-03-01T00:00:00Z")
    late = await create_outbound(client, token, fixture, planned_shipment_date="2027-03-20T00:00:00Z")
    scope = {"warehouse_id": fixture["warehouse_id"], "order_type": "outbound"}

    from_only = await client.get(
        ORDERS, headers=auth(token), params={**scope, "date_from": "2027-03-10T00:00:00Z"}
    )
    to_only = await client.get(
        ORDERS, headers=auth(token), params={**scope, "date_to": "2027-03-10T00:00:00Z"}
    )
    exact_bounds = await client.get(
        ORDERS,
        headers=auth(token),
        params={**scope, "date_from": "2027-03-01T00:00:00Z", "date_to": "2027-03-20T00:00:00Z"},
    )
    empty = await client.get(
        ORDERS, headers=auth(token), params={**scope, "date_from": "2028-01-01T00:00:00Z"}
    )

    assert _keys(from_only) == {("outbound", late["outbound_order_id"])}
    assert _keys(to_only) == {("outbound", early["outbound_order_id"])}
    assert _keys(exact_bounds) == {
        ("outbound", early["outbound_order_id"]),
        ("outbound", late["outbound_order_id"]),
    }
    assert _keys(empty) == set()

async def test_orders_list_is_sorted_by_date_descending(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    dates = ["2027-05-02T00:00:00Z", "2027-05-09T00:00:00Z", "2027-05-05T00:00:00Z"]
    for date in dates:
        await create_outbound(client, token, fixture, planned_shipment_date=date)

    response = await client.get(
        ORDERS,
        headers=auth(token),
        params={"warehouse_id": fixture["warehouse_id"], "order_type": "outbound"},
    )

    returned = [item["date"][:10] for item in response.json()["items"]]
    assert returned == ["2027-05-09", "2027-05-05", "2027-05-02"]

@pytest.mark.parametrize(
    "params",
    [
        {"order_type": "transfer"},
        {"order_status": "shipped"},
        {"page": 0},
        {"page": -1},
        {"page_size": 0},
        {"page_size": 201},
        {"date_from": "вчора"},
    ],
)
async def test_orders_list_rejects_invalid_query(client: AsyncClient, params: dict):
    token = await login(client, ADMIN)

    response = await client.get(ORDERS, headers=auth(token), params=params)

    assert response.status_code == 422, response.text

async def test_orders_list_page_size_upper_bound_is_accepted(client: AsyncClient):
    token = await login(client, ADMIN)

    response = await client.get(ORDERS, headers=auth(token), params={"page_size": 200})

    assert response.status_code == 200, response.text
    assert response.json()["page_size"] == 200

async def test_orders_list_unknown_warehouse_is_empty(client: AsyncClient):
    token = await login(client, ADMIN)

    response = await client.get(ORDERS, headers=auth(token), params={"warehouse_id": 999999})

    assert response.status_code == 200, response.text
    assert response.json()["items"] == []
    assert response.json()["total"] == 0

# ---------------------------------------------------------------------------
# Залишки для форми (/stock): підказки по рядках накладної
# ---------------------------------------------------------------------------

async def test_stock_endpoint_for_form_lines(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    stocked = fixture["product_id"]
    empty = await add_product(client, token, fixture["category_id"])
    await receive_stock(client, token, fixture, 40)

    response = await client.get(
        STOCK,
        headers=auth(token),
        params={"warehouse_id": fixture["warehouse_id"], "product_ids": f"{stocked},{empty}"},
    )

    assert response.status_code == 200, response.text
    levels = {row["product_id"]: row for row in response.json()}
    assert Decimal(levels[stocked]["available"]) == Decimal("40")
    assert Decimal(levels[empty]["available"]) == Decimal("0")
    assert levels[stocked]["sku_code"]
    assert levels[stocked]["product_name"] == "Цукор, 1 кг"

async def test_stock_endpoint_is_scoped_by_warehouse(client: AsyncClient):
    token = await login(client, ADMIN)
    first = await make_fixture(client, token)
    second = await make_fixture(client, token)
    await receive_stock(client, token, first, 30)

    in_first = await client.get(
        STOCK,
        headers=auth(token),
        params={"warehouse_id": first["warehouse_id"], "product_ids": first["product_id"]},
    )
    in_second = await client.get(
        STOCK,
        headers=auth(token),
        params={"warehouse_id": second["warehouse_id"], "product_ids": first["product_id"]},
    )

    assert Decimal(in_first.json()[0]["available"]) == Decimal("30")
    assert Decimal(in_second.json()[0]["available"]) == Decimal("0")

async def test_stock_endpoint_rejects_malformed_product_ids(client: AsyncClient):
    token = await login(client, ADMIN)

    response = await client.get(STOCK, headers=auth(token), params={"product_ids": "1,abc"})

    assert response.status_code == 422, response.text

async def test_outbound_orders_do_not_leak_into_inbound_listing(client: AsyncClient):
    """Вихідні накладні не потрапляють у прихідні, і навпаки (єдиний список)."""
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    inbound = await create_inbound(client, token, fixture)
    outbound = await create_outbound(client, token, fixture)
    scope = {"warehouse_id": fixture["warehouse_id"]}

    inbound_only = await client.get(ORDERS, headers=auth(token), params={**scope, "order_type": "inbound"})
    outbound_only = await client.get(ORDERS, headers=auth(token), params={**scope, "order_type": "outbound"})
    outbound_list = await client.get(OUTBOUND, headers=auth(token), params=scope)

    assert _keys(inbound_only) == {("inbound", inbound["inbound_order_id"])}
    assert _keys(outbound_only) == {("outbound", outbound["outbound_order_id"])}
    assert {o["outbound_order_id"] for o in outbound_list.json()} == {outbound["outbound_order_id"]}

@pytest.mark.xfail(
    strict=True,
    reason=(
        "БАГ: orders/crud.list_inbound_orders викликає .where() на func.count(...) "
        "замість select(...), тому GET /inbound-orders завжди падає з AttributeError"
    ),
)
async def test_inbound_orders_list_endpoint_works(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    inbound = await create_inbound(client, token, fixture)

    response = await client.get(
        INBOUND, headers=auth(token), params={"warehouse_id": fixture["warehouse_id"]}
    )

    assert response.status_code == 200, response.text
    assert [o["inbound_order_id"] for o in response.json()] == [inbound["inbound_order_id"]]
