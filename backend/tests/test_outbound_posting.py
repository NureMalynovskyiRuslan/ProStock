"""Проведення накладних: списання залишків, помилки нестачі, граничні випадки."""
from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest
from httpx import AsyncClient
from sqlalchemy import delete

from app.core.db import async_session_maker
from app.core.enums import StockMovementType
from app.orders.models import OutboundOrderItem
from tests.invoice_utils import (
    cleanup_orders,  # noqa: F401 - фікстура
    INBOUND,
    OUTBOUND,
    SHELF_CAPACITY,
    add_product,
    available,
    create_inbound,
    create_outbound,
    get_outbound,
    line,
    make_fixture,
    movements,
    outbound_payload,
    post_outbound,
    receive_stock,
    stock_rows,
    user_id_by_email,
)
from tests.utils import (
    ADMIN,
    MANAGER,
    WORKER,
    auth,
    create_warehouse_with_shelves,
    login,
)

pytestmark = pytest.mark.usefixtures("cleanup_orders")

def _by_product(shortages: list[dict]) -> dict[int, dict]:
    return {item["product_id"]: item for item in shortages}

# ---------------------------------------------------------------------------
# Успішне проведення
# ---------------------------------------------------------------------------

async def test_post_returns_summary_and_completes_order(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    await receive_stock(client, token, fixture, 100)
    second = await add_product(client, token, fixture["category_id"])
    await receive_stock(client, token, fixture, 50, product_id=second)
    order = await create_outbound(
        client, token, fixture, [line(fixture["product_id"], 30), line(second, "7.5")]
    )

    response = await post_outbound(client, token, order["outbound_order_id"])

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["order_id"] == order["outbound_order_id"]
    assert body["order_type"] == "outbound"
    assert body["order_status"] == "completed"
    assert body["items_count"] == 2
    assert Decimal(body["total_quantity"]) == Decimal("37.5")
    assert Decimal(body["processed_quantity"]) == Decimal("37.5")
    assert body["posted_at"]
    assert (await get_outbound(client, token, order["outbound_order_id"]))["order_status"] == "completed"

async def test_post_decreases_stock_for_every_line(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    second = await add_product(client, token, fixture["category_id"])
    await receive_stock(client, token, fixture, 100)
    await receive_stock(client, token, fixture, 40, product_id=second)
    order = await create_outbound(
        client, token, fixture, [line(fixture["product_id"], 25), line(second, 15)]
    )

    assert (await post_outbound(client, token, order["outbound_order_id"])).status_code == 200

    assert await available(client, token, fixture) == Decimal("75")
    assert await available(client, token, fixture, second) == Decimal("25")

async def test_post_writes_negative_stock_movements(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    await receive_stock(client, token, fixture, 100)
    order = await create_outbound(client, token, fixture, [line(fixture["product_id"], 30)])

    assert (await post_outbound(client, token, order["outbound_order_id"])).status_code == 200

    outgoing = await movements(fixture["warehouse_id"], StockMovementType.OUTBOUND_ORDER)
    assert len(outgoing) == 1
    movement = outgoing[0]
    assert movement.quantity == Decimal("-30")
    assert movement.product_id == fixture["product_id"]
    assert movement.order_id == order["outbound_order_id"]
    assert movement.created_by_user_id == await user_id_by_email(ADMIN[0])

    incoming = await movements(fixture["warehouse_id"], StockMovementType.INBOUND_ORDER)
    assert sum(m.quantity for m in incoming) + sum(m.quantity for m in outgoing) == Decimal("70")

async def test_post_by_manager_records_manager_in_movements(client: AsyncClient):
    admin = await login(client, ADMIN)
    manager = await login(client, MANAGER)
    fixture = await make_fixture(client, admin)
    await receive_stock(client, admin, fixture, 10)
    order = await create_outbound(client, manager, fixture, [line(fixture["product_id"], 4)])

    assert (await post_outbound(client, manager, order["outbound_order_id"])).status_code == 200

    outgoing = await movements(fixture["warehouse_id"], StockMovementType.OUTBOUND_ORDER)
    assert {m.created_by_user_id for m in outgoing} == {await user_id_by_email(MANAGER[0])}

async def test_post_leaves_other_products_and_warehouses_untouched(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    other_warehouse = await make_fixture(client, token)
    bystander = await add_product(client, token, fixture["category_id"])
    await receive_stock(client, token, fixture, 60)
    await receive_stock(client, token, fixture, 11, product_id=bystander)
    await receive_stock(client, token, other_warehouse, 80)
    order = await create_outbound(client, token, fixture, [line(fixture["product_id"], 20)])

    assert (await post_outbound(client, token, order["outbound_order_id"])).status_code == 200

    assert await available(client, token, fixture, bystander) == Decimal("11")
    assert await available(client, token, other_warehouse) == Decimal("80")

# ---------------------------------------------------------------------------
# Граничні випадки залишків
# ---------------------------------------------------------------------------

async def test_post_exact_stock_removes_all_rows(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    await receive_stock(client, token, fixture, 12)
    order = await create_outbound(client, token, fixture, [line(fixture["product_id"], 12)])

    response = await post_outbound(client, token, order["outbound_order_id"])

    assert response.status_code == 200, response.text
    assert await available(client, token, fixture) == Decimal("0")
    assert await stock_rows(fixture["warehouse_id"], fixture["product_id"]) == []

async def test_post_one_hundredth_over_stock_is_rejected(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    await receive_stock(client, token, fixture, 12)
    order = await create_outbound(client, token, fixture, [line(fixture["product_id"], "12.01")])

    response = await post_outbound(client, token, order["outbound_order_id"])

    assert response.status_code == 409, response.text
    shortage = response.json()["detail"]["shortages"][0]
    assert Decimal(shortage["available"]) == Decimal("12")
    assert Decimal(shortage["shortage"]) == Decimal("0.01")
    assert await available(client, token, fixture) == Decimal("12")

async def test_post_minimal_quantity_leaves_remainder(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    await receive_stock(client, token, fixture, "5.5")
    order = await create_outbound(client, token, fixture, [line(fixture["product_id"], "0.01")])

    response = await post_outbound(client, token, order["outbound_order_id"])

    assert response.status_code == 200, response.text
    assert await available(client, token, fixture) == Decimal("5.49")

async def test_post_fractional_quantities_deplete_stock_exactly(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    await receive_stock(client, token, fixture, "10.10")
    first = await create_outbound(client, token, fixture, [line(fixture["product_id"], "3.33")])
    second = await create_outbound(client, token, fixture, [line(fixture["product_id"], "6.77")])

    assert (await post_outbound(client, token, first["outbound_order_id"])).status_code == 200
    assert await available(client, token, fixture) == Decimal("6.77")
    assert (await post_outbound(client, token, second["outbound_order_id"])).status_code == 200

    assert await available(client, token, fixture) == Decimal("0")
    assert await stock_rows(fixture["warehouse_id"]) == []

async def test_post_takes_stock_from_several_shelves(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    await receive_stock(client, token, fixture, SHELF_CAPACITY + 200)
    rows = await stock_rows(fixture["warehouse_id"], fixture["product_id"])
    assert sorted(r.quantity for r in rows) == [Decimal("200"), Decimal(SHELF_CAPACITY)]
    order = await create_outbound(client, token, fixture, [line(fixture["product_id"], 600)])

    response = await post_outbound(client, token, order["outbound_order_id"])

    assert response.status_code == 200, response.text
    assert await available(client, token, fixture) == Decimal("100")
    remaining = await stock_rows(fixture["warehouse_id"], fixture["product_id"])
    assert [r.quantity for r in remaining] == [Decimal("100")]
    outgoing = await movements(fixture["warehouse_id"], StockMovementType.OUTBOUND_ORDER)
    assert len(outgoing) == 2
    assert sum(m.quantity for m in outgoing) == Decimal("-600")
    assert {m.shelf_id for m in outgoing} == set(fixture["shelf_ids"])

async def test_stock_in_different_package_types_is_summed(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    boxes = await create_inbound(client, token, fixture, [line(fixture["product_id"], 30, "box")])
    pallets = await create_inbound(client, token, fixture, [line(fixture["product_id"], 20, "pallet")])
    for inbound in (boxes, pallets):
        posted = await client.post(
            f"{INBOUND}/{inbound['inbound_order_id']}/post", headers=auth(token)
        )
        assert posted.status_code == 200, posted.text
    order = await create_outbound(client, token, fixture, [line(fixture["product_id"], 45, "box")])

    response = await post_outbound(client, token, order["outbound_order_id"])

    assert response.status_code == 200, response.text
    assert await available(client, token, fixture) == Decimal("5")

async def test_post_sequential_orders_deplete_stock_then_fail(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    await receive_stock(client, token, fixture, 100)
    first = await create_outbound(client, token, fixture, [line(fixture["product_id"], 60)])
    second = await create_outbound(client, token, fixture, [line(fixture["product_id"], 60)])

    assert (await post_outbound(client, token, first["outbound_order_id"])).status_code == 200
    response = await post_outbound(client, token, second["outbound_order_id"])

    assert response.status_code == 409, response.text
    shortage = response.json()["detail"]["shortages"][0]
    assert Decimal(shortage["requested"]) == Decimal("60")
    assert Decimal(shortage["available"]) == Decimal("40")
    assert Decimal(shortage["shortage"]) == Decimal("20")
    assert (await get_outbound(client, token, second["outbound_order_id"]))["order_status"] == "draft"
    assert await available(client, token, fixture) == Decimal("40")

# ---------------------------------------------------------------------------
# Помилки нестачі залишків (те, що показує форма накладної)
# ---------------------------------------------------------------------------

async def test_shortage_error_has_message_and_detailed_rows(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    await receive_stock(client, token, fixture, 10)
    order = await create_outbound(client, token, fixture, [line(fixture["product_id"], 25)])

    response = await post_outbound(client, token, order["outbound_order_id"])

    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert "залишків" in detail["message"].lower()
    assert len(detail["shortages"]) == 1
    row = detail["shortages"][0]
    assert set(row) == {"product_id", "product_name", "sku_code", "requested", "available", "shortage"}
    assert row["product_id"] == fixture["product_id"]
    assert row["product_name"] == "Цукор, 1 кг"
    assert row["sku_code"]
    assert Decimal(row["requested"]) == Decimal("25")
    assert Decimal(row["available"]) == Decimal("10")
    assert Decimal(row["shortage"]) == Decimal("15")

async def test_shortage_lists_only_short_products(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    enough = fixture["product_id"]
    short = await add_product(client, token, fixture["category_id"], name="Борошно")
    missing = await add_product(client, token, fixture["category_id"], name="Сіль")
    await receive_stock(client, token, fixture, 100, product_id=enough)
    await receive_stock(client, token, fixture, 5, product_id=short)
    order = await create_outbound(
        client,
        token,
        fixture,
        [line(enough, 50), line(short, 9), line(missing, 3)],
    )

    response = await post_outbound(client, token, order["outbound_order_id"])

    assert response.status_code == 409, response.text
    rows = _by_product(response.json()["detail"]["shortages"])
    assert set(rows) == {short, missing}
    assert rows[short]["product_name"] == "Борошно"
    assert Decimal(rows[short]["shortage"]) == Decimal("4")
    assert rows[missing]["product_name"] == "Сіль"
    assert Decimal(rows[missing]["available"]) == Decimal("0")
    assert Decimal(rows[missing]["shortage"]) == Decimal("3")

async def test_shortage_rolls_back_whole_order(client: AsyncClient):
    """Якщо хоч одного товару бракує - нічого не списується."""
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    short = await add_product(client, token, fixture["category_id"])
    await receive_stock(client, token, fixture, 100)
    await receive_stock(client, token, fixture, 1, product_id=short)
    order = await create_outbound(
        client, token, fixture, [line(fixture["product_id"], 40), line(short, 2)]
    )

    response = await post_outbound(client, token, order["outbound_order_id"])

    assert response.status_code == 409, response.text
    assert await available(client, token, fixture) == Decimal("100")
    assert await available(client, token, fixture, short) == Decimal("1")
    assert await movements(fixture["warehouse_id"], StockMovementType.OUTBOUND_ORDER) == []
    assert (await get_outbound(client, token, order["outbound_order_id"]))["order_status"] == "draft"

async def test_same_product_in_several_lines_is_summed_in_shortage(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    await receive_stock(client, token, fixture, 40)
    order = await create_outbound(
        client,
        token,
        fixture,
        [line(fixture["product_id"], 30, "box"), line(fixture["product_id"], 30, "pallet")],
    )

    response = await post_outbound(client, token, order["outbound_order_id"])

    assert response.status_code == 409, response.text
    rows = response.json()["detail"]["shortages"]
    assert len(rows) == 1, "UI показує по одному рядку на товар"
    assert Decimal(rows[0]["requested"]) == Decimal("60")
    assert Decimal(rows[0]["shortage"]) == Decimal("20")

async def test_same_product_in_several_lines_within_stock_is_posted(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    await receive_stock(client, token, fixture, 60)
    order = await create_outbound(
        client,
        token,
        fixture,
        [line(fixture["product_id"], 30, "box"), line(fixture["product_id"], 30, "pallet")],
    )

    response = await post_outbound(client, token, order["outbound_order_id"])

    assert response.status_code == 200, response.text
    assert Decimal(response.json()["total_quantity"]) == Decimal("60")
    assert await available(client, token, fixture) == Decimal("0")

async def test_stock_of_another_warehouse_does_not_cover_shortage(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    other = await make_fixture(client, token)
    await receive_stock(client, token, other, 500, product_id=fixture["product_id"])
    order = await create_outbound(client, token, fixture, [line(fixture["product_id"], 5)])

    response = await post_outbound(client, token, order["outbound_order_id"])

    assert response.status_code == 409, response.text
    assert Decimal(response.json()["detail"]["shortages"][0]["available"]) == Decimal("0")
    assert await available(client, token, other, fixture["product_id"]) == Decimal("500")

async def test_post_succeeds_after_stock_is_replenished(client: AsyncClient):
    """Сценарій кнопки «Провести»: помилка, поповнення, повторна спроба."""
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    order = await create_outbound(client, token, fixture, [line(fixture["product_id"], 30)])
    order_id = order["outbound_order_id"]

    first_try = await post_outbound(client, token, order_id)
    assert first_try.status_code == 409, first_try.text

    await receive_stock(client, token, fixture, 30)
    second_try = await post_outbound(client, token, order_id)

    assert second_try.status_code == 200, second_try.text
    assert (await get_outbound(client, token, order_id))["order_status"] == "completed"
    assert await available(client, token, fixture) == Decimal("0")

async def test_post_succeeds_after_order_is_edited_to_fit_stock(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    await receive_stock(client, token, fixture, 10)
    order = await create_outbound(client, token, fixture, [line(fixture["product_id"], 50)])
    order_id = order["outbound_order_id"]
    assert (await post_outbound(client, token, order_id)).status_code == 409

    patched = await client.patch(
        f"{OUTBOUND}/{order_id}",
        headers=auth(token),
        json={"items": [line(fixture["product_id"], 10)]},
    )
    assert patched.status_code == 200, patched.text

    assert (await post_outbound(client, token, order_id)).status_code == 200
    assert await available(client, token, fixture) == Decimal("0")

# ---------------------------------------------------------------------------
# Перевірка залишків без проведення (/availability)
# ---------------------------------------------------------------------------

async def test_availability_fits_with_exact_stock(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    await receive_stock(client, token, fixture, 25)
    order = await create_outbound(client, token, fixture, [line(fixture["product_id"], 25)])

    response = await client.get(
        f"{OUTBOUND}/{order['outbound_order_id']}/availability", headers=auth(token)
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["fits"] is True
    assert body["shortages"] == []
    assert Decimal(body["available"][str(fixture["product_id"])]) == Decimal("25")

async def test_availability_reports_shortage_without_changing_anything(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    await receive_stock(client, token, fixture, 25)
    order = await create_outbound(client, token, fixture, [line(fixture["product_id"], "25.01")])

    response = await client.get(
        f"{OUTBOUND}/{order['outbound_order_id']}/availability", headers=auth(token)
    )

    body = response.json()
    assert body["fits"] is False
    assert Decimal(body["shortages"][0]["shortage"]) == Decimal("0.01")
    assert await available(client, token, fixture) == Decimal("25")
    assert (await get_outbound(client, token, order["outbound_order_id"]))["order_status"] == "draft"

async def test_availability_without_any_stock(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    order = await create_outbound(client, token, fixture, [line(fixture["product_id"], 3)])

    response = await client.get(
        f"{OUTBOUND}/{order['outbound_order_id']}/availability", headers=auth(token)
    )

    body = response.json()
    assert body["fits"] is False
    assert Decimal(body["shortages"][0]["available"]) == Decimal("0")
    assert body["available"] == {}

async def test_availability_agrees_with_post_result(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    await receive_stock(client, token, fixture, 20)
    fits = await create_outbound(client, token, fixture, [line(fixture["product_id"], 20)])
    too_much = await create_outbound(client, token, fixture, [line(fixture["product_id"], 21)])

    for order, expected_fits in ((too_much, False), (fits, True)):
        check = await client.get(
            f"{OUTBOUND}/{order['outbound_order_id']}/availability", headers=auth(token)
        )
        posted = await post_outbound(client, token, order["outbound_order_id"])
        assert check.json()["fits"] is expected_fits
        assert (posted.status_code == 200) is expected_fits

# ---------------------------------------------------------------------------
# Стани накладної й повторне проведення
# ---------------------------------------------------------------------------

async def test_post_cannot_be_repeated_and_does_not_write_off_twice(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    await receive_stock(client, token, fixture, 100)
    order = await create_outbound(client, token, fixture, [line(fixture["product_id"], 30)])
    order_id = order["outbound_order_id"]

    first = await post_outbound(client, token, order_id)
    second = await post_outbound(client, token, order_id)

    assert first.status_code == 200, first.text
    assert second.status_code == 409, second.text
    assert "чернет" in second.json()["detail"].lower()
    assert await available(client, token, fixture) == Decimal("70")
    outgoing = await movements(fixture["warehouse_id"], StockMovementType.OUTBOUND_ORDER)
    assert len(outgoing) == 1

@pytest.mark.parametrize("order_status", ["in_progress", "ready", "completed", "canceled"])
async def test_only_draft_can_be_posted(client: AsyncClient, order_status: str):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    await receive_stock(client, token, fixture, 100)
    order = await create_outbound(
        client,
        token,
        fixture,
        [line(fixture["product_id"], 10)],
        order_status=order_status,
    )

    response = await post_outbound(client, token, order["outbound_order_id"])

    assert response.status_code == 409, response.text
    assert order_status in response.json()["detail"]
    assert await available(client, token, fixture) == Decimal("100")
    assert (await get_outbound(client, token, order["outbound_order_id"]))["order_status"] == order_status

async def test_post_unknown_order_returns_404(client: AsyncClient):
    token = await login(client, ADMIN)

    response = await post_outbound(client, token, 999999)

    assert response.status_code == 404, response.text

async def test_storekeeper_cannot_post(client: AsyncClient):
    admin = await login(client, ADMIN)
    worker = await login(client, WORKER)
    fixture = await make_fixture(client, admin)
    await receive_stock(client, admin, fixture, 10)
    order = await create_outbound(client, admin, fixture, [line(fixture["product_id"], 5)])

    response = await post_outbound(client, worker, order["outbound_order_id"])

    assert response.status_code == 403, response.text
    assert await available(client, admin, fixture) == Decimal("10")

async def test_draft_without_lines_cannot_be_posted(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    order = await create_outbound(client, token, fixture)
    order_id = order["outbound_order_id"]
    # Рядки видаляємо напряму в БД, щоб тест не залежав від валідації PATCH.
    async with async_session_maker() as session:
        await session.execute(
            delete(OutboundOrderItem).where(OutboundOrderItem.outbound_order_id == order_id)
        )
        await session.commit()

    response = await post_outbound(client, token, order_id)

    assert response.status_code == 409, response.text
    assert "без позицій" in response.json()["detail"]
    assert (await get_outbound(client, token, order_id))["order_status"] == "draft"

# ---------------------------------------------------------------------------
# Відомі проблеми: тести падають, поки код не виправлено (xfail strict)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    strict=True,
    reason=(
        "БАГ: OutboundOrderUpdate.items не має min_length=1 (на відміну від "
        "OutboundOrderCreate), тому PATCH з items=[] залишає накладну без рядків"
    ),
)
async def test_update_cannot_remove_all_lines(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    order = await create_outbound(client, token, fixture)

    response = await client.patch(
        f"{OUTBOUND}/{order['outbound_order_id']}", headers=auth(token), json={"items": []}
    )

    assert response.status_code == 422, response.text

@pytest.mark.xfail(
    strict=True,
    reason=(
        "БАГ: PATCH дозволяє змінювати вже проведену (completed) накладну; "
        "update_outbound_order не перевіряє статус"
    ),
)
async def test_completed_order_cannot_be_edited(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    await receive_stock(client, token, fixture, 100)
    order = await create_outbound(client, token, fixture, [line(fixture["product_id"], 10)])
    order_id = order["outbound_order_id"]
    assert (await post_outbound(client, token, order_id)).status_code == 200

    response = await client.patch(
        f"{OUTBOUND}/{order_id}",
        headers=auth(token),
        json={"items": [line(fixture["product_id"], 50)]},
    )

    assert response.status_code in (400, 409, 422), response.text
    assert Decimal((await get_outbound(client, token, order_id))["items"][0]["quantity"]) == Decimal("10")

@pytest.mark.xfail(
    strict=True,
    reason=(
        "БАГ: PATCH дозволяє повернути проведену накладну в draft і провести її "
        "вдруге - залишки списуються двічі, а рух по складу не відкочується"
    ),
)
async def test_posted_order_cannot_be_reopened_and_posted_again(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    await receive_stock(client, token, fixture, 100)
    order = await create_outbound(client, token, fixture, [line(fixture["product_id"], 10)])
    order_id = order["outbound_order_id"]
    assert (await post_outbound(client, token, order_id)).status_code == 200

    await client.patch(f"{OUTBOUND}/{order_id}", headers=auth(token), json={"order_status": "draft"})
    await post_outbound(client, token, order_id)

    assert await available(client, token, fixture) == Decimal("90")

@pytest.mark.xfail(
    strict=True,
    reason=(
        "БАГ: POST /outbound-orders приймає order_status=completed, тож накладна "
        "стає «Завершено» без проведення й без списання залишків"
    ),
)
async def test_order_cannot_be_created_as_completed_without_posting(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    await receive_stock(client, token, fixture, 100)

    response = await client.post(
        OUTBOUND,
        headers=auth(token),
        json=outbound_payload(
            fixture, [line(fixture["product_id"], 10)], order_status="completed"
        ),
    )

    assert response.status_code >= 400 or response.json()["order_status"] == "draft", response.text

@pytest.mark.xfail(
    strict=False,
    reason=(
        "БАГ (гонка): проведення не блокує рядки залишків (немає SELECT ... FOR UPDATE). "
        "Паралельні запити можуть списати залишок двічі або продати більше, ніж є. "
        "Результат залежить від таймінгу, тому strict=False"
    ),
)
async def test_parallel_posts_of_same_order_write_off_once(client: AsyncClient):
    token = await login(client, ADMIN)
    for _ in range(5):
        fixture = await make_fixture(client, token)
        await receive_stock(client, token, fixture, 100)
        order = await create_outbound(client, token, fixture, [line(fixture["product_id"], 30)])

        responses = await asyncio.gather(
            *[post_outbound(client, token, order["outbound_order_id"]) for _ in range(2)]
        )

        assert sorted(r.status_code for r in responses) == [200, 409]
        assert await available(client, token, fixture) == Decimal("70")

@pytest.mark.xfail(
    strict=False,
    reason=(
        "БАГ (гонка): дві різні накладні на той самий залишок можуть пройти одночасно. "
        "Результат залежить від таймінгу, тому strict=False"
    ),
)
async def test_parallel_posts_cannot_oversell_stock(client: AsyncClient):
    token = await login(client, ADMIN)
    for _ in range(5):
        fixture = await make_fixture(client, token)
        await receive_stock(client, token, fixture, 100)
        first = await create_outbound(client, token, fixture, [line(fixture["product_id"], 60)])
        second = await create_outbound(client, token, fixture, [line(fixture["product_id"], 60)])

        responses = await asyncio.gather(
            post_outbound(client, token, first["outbound_order_id"]),
            post_outbound(client, token, second["outbound_order_id"]),
        )

        assert sorted(r.status_code for r in responses) == [200, 409]
        assert await available(client, token, fixture) == Decimal("40")

# ---------------------------------------------------------------------------
# Проведення прихідних накладних (залишки, з яких списує вихідна)
# ---------------------------------------------------------------------------

async def test_post_inbound_writes_positive_movements(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)

    await receive_stock(client, token, fixture, 70)

    incoming = await movements(fixture["warehouse_id"], StockMovementType.INBOUND_ORDER)
    assert sum(m.quantity for m in incoming) == Decimal("70")
    assert all(m.quantity > 0 for m in incoming)

async def test_inbound_exactly_fills_shelf_then_overflow_is_rejected(client: AsyncClient):
    token = await login(client, ADMIN)
    warehouse = await create_warehouse_with_shelves(
        [{"code": "TINY", "width": "2", "length": "1", "height": "1", "max_weight": "10"}]
    )
    fixture = {**(await make_fixture(client, token)), **warehouse}

    exact = await create_inbound(client, token, fixture, [line(fixture["product_id"], 10)])
    posted = await client.post(f"{INBOUND}/{exact['inbound_order_id']}/post", headers=auth(token))
    assert posted.status_code == 200, posted.text
    assert await available(client, token, fixture) == Decimal("10")

    overflow = await create_inbound(client, token, fixture, [line(fixture["product_id"], "0.01")])
    rejected = await client.post(f"{INBOUND}/{overflow['inbound_order_id']}/post", headers=auth(token))

    assert rejected.status_code == 409, rejected.text
    assert await available(client, token, fixture) == Decimal("10")

async def test_inbound_partial_fit_is_rolled_back(client: AsyncClient):
    token = await login(client, ADMIN)
    warehouse = await create_warehouse_with_shelves(
        [{"code": "SMALL", "width": "2", "length": "1", "height": "1", "max_weight": "10"}]
    )
    fixture = {**(await make_fixture(client, token)), **warehouse}
    inbound = await create_inbound(client, token, fixture, [line(fixture["product_id"], "10.5")])

    response = await client.post(
        f"{INBOUND}/{inbound['inbound_order_id']}/post", headers=auth(token)
    )

    assert response.status_code == 409, response.text
    row = response.json()["detail"]["shortages"][0]
    assert Decimal(row["placed"]) == Decimal("10")
    assert Decimal(row["unplaced"]) == Decimal("0.5")
    assert await available(client, token, fixture) == Decimal("0")
    assert await stock_rows(fixture["warehouse_id"]) == []
    assert await movements(fixture["warehouse_id"], StockMovementType.INBOUND_ORDER) == []

async def test_inbound_to_warehouse_without_active_shelves_is_rejected(client: AsyncClient):
    token = await login(client, ADMIN)
    warehouse = await create_warehouse_with_shelves(
        [{"code": "OFF", "status": "inactive", "max_weight": "500"}]
    )
    fixture = {**(await make_fixture(client, token)), **warehouse}
    inbound = await create_inbound(client, token, fixture, [line(fixture["product_id"], 1)])

    response = await client.post(
        f"{INBOUND}/{inbound['inbound_order_id']}/post", headers=auth(token)
    )

    assert response.status_code == 409, response.text
    assert "активної полиці" in response.json()["detail"]["shortages"][0]["reason"]

@pytest.mark.parametrize("order_status", ["in_progress", "ready", "completed", "canceled"])
async def test_only_draft_inbound_can_be_posted(client: AsyncClient, order_status: str):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    inbound = await create_inbound(
        client, token, fixture, [line(fixture["product_id"], 5)], order_status=order_status
    )

    response = await client.post(
        f"{INBOUND}/{inbound['inbound_order_id']}/post", headers=auth(token)
    )

    assert response.status_code == 409, response.text
    assert await available(client, token, fixture) == Decimal("0")
