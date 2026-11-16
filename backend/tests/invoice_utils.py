"""Спільні хелпери для тестів накладних (прихідних і вихідних)."""
from __future__ import annotations

from decimal import Decimal

import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import delete, func, select

from app.core.db import async_session_maker
from app.core.enums import StockMovementType
from app.orders.models import InboundOrder, OutboundOrder
from app.stock.models import StockItem, StockMovement
from app.users.models import User
from tests.utils import (
    API,
    auth,
    create_category,
    create_product,
    create_recipient,
    create_supplier,
    create_warehouse_with_shelves,
    unique,
)

INBOUND = f"{API}/inbound-orders"
OUTBOUND = f"{API}/outbound-orders"
ORDERS = f"{API}/orders"
STOCK = f"{API}/stock"

# Товар 0.2 x 0.2 x 0.1 м і 1 кг -> 0.004 м3. Полиця 2 x 1 x 1 м / 500 кг вміщує 500 шт.
SHELF_CAPACITY = 500
TWO_SHELVES = [
    {"code": "A-1", "width": "2", "length": "1", "height": "1", "max_weight": "500"},
    {"code": "A-2", "width": "2", "length": "1", "height": "1", "max_weight": "500"},
]

PLANNED_DATE = "2026-12-02T00:00:00Z"

async def make_fixture(
    client: AsyncClient,
    token: str,
    shelves: list[dict] | None = None,
) -> dict:
    """Новий склад із полицями, категорія, товар, постачальник і одержувач."""
    warehouse = await create_warehouse_with_shelves(shelves or TWO_SHELVES)
    category_id = await create_category(client, token)
    product_id = await add_product(client, token, category_id)
    return {
        **warehouse,
        "category_id": category_id,
        "product_id": product_id,
        "supplier_id": await create_supplier(client, token),
        "recipient_id": await create_recipient(client, token),
    }

async def add_product(
    client: AsyncClient,
    token: str,
    category_id: int,
    **overrides,
) -> int:
    payload = {
        "name": "Цукор, 1 кг",
        "width": 0.2,
        "length": 0.2,
        "height": 0.1,
        "weight": 1.0,
    }
    payload.update(overrides)
    return await create_product(client, token, category_id, **payload)

def line(product_id: int, quantity: str | int | Decimal, package_type: str = "box") -> dict:
    return {
        "product_id": product_id,
        "quantity": str(quantity),
        "package_type": package_type,
    }

def outbound_payload(fixture: dict, items: list[dict] | None = None, **overrides) -> dict:
    payload = {
        "recipient_id": fixture["recipient_id"],
        "warehouse_id": fixture["warehouse_id"],
        "planned_shipment_date": PLANNED_DATE,
        "items": items if items is not None else [line(fixture["product_id"], 10)],
    }
    payload.update(overrides)
    return payload

async def create_outbound(
    client: AsyncClient,
    token: str,
    fixture: dict,
    items: list[dict] | None = None,
    **overrides,
) -> dict:
    response = await client.post(
        OUTBOUND,
        headers=auth(token),
        json=outbound_payload(fixture, items, **overrides),
    )
    assert response.status_code == 201, response.text
    return response.json()

async def create_inbound(
    client: AsyncClient,
    token: str,
    fixture: dict,
    items: list[dict] | None = None,
    **overrides,
) -> dict:
    payload = {
        "supplier_id": fixture["supplier_id"],
        "warehouse_id": fixture["warehouse_id"],
        "invoice_number": unique("INV"),
        "expected_delivery_date": "2026-12-01T00:00:00Z",
        "items": items if items is not None else [line(fixture["product_id"], 100)],
    }
    payload.update(overrides)
    response = await client.post(INBOUND, headers=auth(token), json=payload)
    assert response.status_code == 201, response.text
    return response.json()

async def receive_stock(
    client: AsyncClient,
    token: str,
    fixture: dict,
    quantity: str | int | Decimal = 100,
    product_id: int | None = None,
) -> None:
    """Створює й проводить прихідну накладну: на складі з'являється залишок."""
    inbound = await create_inbound(
        client,
        token,
        fixture,
        [line(product_id or fixture["product_id"], quantity)],
    )
    response = await client.post(
        f"{INBOUND}/{inbound['inbound_order_id']}/post", headers=auth(token)
    )
    assert response.status_code == 200, response.text

async def post_outbound(client: AsyncClient, token: str, order_id: int):
    return await client.post(f"{OUTBOUND}/{order_id}/post", headers=auth(token))

async def get_outbound(client: AsyncClient, token: str, order_id: int) -> dict:
    response = await client.get(f"{OUTBOUND}/{order_id}", headers=auth(token))
    assert response.status_code == 200, response.text
    return response.json()

async def available(
    client: AsyncClient,
    token: str,
    fixture: dict,
    product_id: int | None = None,
) -> Decimal:
    """Залишок товару на складі фікстури за відповіддю API."""
    pid = product_id or fixture["product_id"]
    response = await client.get(
        STOCK,
        headers=auth(token),
        params={"warehouse_id": fixture["warehouse_id"], "product_ids": pid},
    )
    assert response.status_code == 200, response.text
    levels = response.json()
    return Decimal(levels[0]["available"]) if levels else Decimal("0")

async def stock_rows(warehouse_id: int, product_id: int | None = None) -> list[StockItem]:
    async with async_session_maker() as session:
        stmt = select(StockItem).where(StockItem.warehouse_id == warehouse_id)
        if product_id is not None:
            stmt = stmt.where(StockItem.product_id == product_id)
        return list((await session.scalars(stmt.order_by(StockItem.shelf_id))).all())

async def movements(
    warehouse_id: int,
    movement_type: StockMovementType,
    product_id: int | None = None,
) -> list[StockMovement]:
    async with async_session_maker() as session:
        stmt = select(StockMovement).where(
            StockMovement.warehouse_id == warehouse_id,
            StockMovement.movement_type == movement_type,
        )
        if product_id is not None:
            stmt = stmt.where(StockMovement.product_id == product_id)
        return list((await session.scalars(stmt.order_by(StockMovement.id))).all())

async def user_id_by_email(email: str) -> int:
    async with async_session_maker() as session:
        return await session.scalar(select(User.id).where(User.email == email))


@pytest_asyncio.fixture(scope="module", loop_scope="session")
async def cleanup_orders():
    """Видаляє накладні, створені тестами модуля.

    БД спільна для всієї сесії, а деякі тести читають першу сторінку
    загального списку /orders, тому сотні накладних звідси їм заважали б.
    """
    async with async_session_maker() as session:
        outbound_start = await session.scalar(select(func.coalesce(func.max(OutboundOrder.id), 0)))
        inbound_start = await session.scalar(select(func.coalesce(func.max(InboundOrder.id), 0)))
    yield
    async with async_session_maker() as session:
        await session.execute(delete(OutboundOrder).where(OutboundOrder.id > outbound_start))
        await session.execute(delete(InboundOrder).where(InboundOrder.id > inbound_start))
        await session.commit()
