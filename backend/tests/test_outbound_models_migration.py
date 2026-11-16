"""Моделі вихідних накладних у БД: обмеження, каскади та міграції alembic."""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest
import pytest_asyncio
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from httpx import AsyncClient
from sqlalchemy import delete, func, inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import settings
from app.core.db import Base, async_session_maker, engine
from app.core.enums import OrderStatus, StockMovementType
from app.orders.models import OutboundOrder, OutboundOrderItem
from app.products.models import Product, Recipient
from app.stock.models import StockItem, StockMovement
from app.warehouses.models import Warehouse
from tests.invoice_utils import (
    cleanup_orders,  # noqa: F401 - фікстура
    PLANNED_DATE,
    create_outbound,
    line,
    make_fixture,
    user_id_by_email,
)
from tests.utils import ADMIN, login, unique

BACKEND_DIR = Path(__file__).resolve().parent.parent
SCRATCH_DB = "prostock_migration_check"
INVOICE_TABLES = {"outbound_order", "outbound_order_item", "stock_item", "stock_movement"}
PREVIOUS_REVISION = "212f6b08f028"  # остання ревізія перед додаванням warehouse_id

pytestmark = pytest.mark.usefixtures("cleanup_orders")

# ---------------------------------------------------------------------------
# Допоміжне
# ---------------------------------------------------------------------------

async def _insert_order(fixture: dict, **overrides) -> int:
    data = {
        "recipient_id": fixture["recipient_id"],
        "warehouse_id": fixture["warehouse_id"],
        "invoice_number": unique("DB"),
        "planned_shipment_date": __import__("datetime").datetime.fromisoformat(
            PLANNED_DATE.replace("Z", "+00:00")
        ),
        "created_by_user_id": await user_id_by_email(ADMIN[0]),
    }
    data.update(overrides)
    async with async_session_maker() as session:
        order = OutboundOrder(**data)
        session.add(order)
        await session.commit()
        return order.id

async def _insert_item(order_id: int, product_id: int, quantity, package_type="box") -> None:
    async with async_session_maker() as session:
        session.add(
            OutboundOrderItem(
                outbound_order_id=order_id,
                product_id=product_id,
                quantity=Decimal(str(quantity)),
                package_type=package_type,
            )
        )
        await session.commit()

async def _count(model, *conditions) -> int:
    async with async_session_maker() as session:
        return await session.scalar(select(func.count()).select_from(model).where(*conditions))

# ---------------------------------------------------------------------------
# Схема таблиць (за моделями)
# ---------------------------------------------------------------------------

async def _inspect(callback):
    async with engine.connect() as connection:
        return await connection.run_sync(lambda sync: callback(inspect(sync)))

async def test_outbound_tables_have_expected_columns(client: AsyncClient):
    columns = await _inspect(
        lambda i: {
            table: {c["name"]: c for c in i.get_columns(table)}
            for table in ("outbound_order", "outbound_order_item")
        }
    )

    order = columns["outbound_order"]
    assert set(order) == {
        "outbound_order_id",
        "recipient_id",
        "warehouse_id",
        "invoice_number",
        "planned_shipment_date",
        "order_status",
        "created_by_user_id",
        "created_at",
    }
    assert all(not column["nullable"] for column in order.values())
    assert order["created_at"]["default"] is not None

    item = columns["outbound_order_item"]
    assert set(item) == {
        "outbound_order_item_id",
        "outbound_order_id",
        "product_id",
        "quantity",
        "package_type",
    }
    assert all(not column["nullable"] for column in item.values())
    assert item["quantity"]["type"].precision == 10
    assert item["quantity"]["type"].scale == 2

async def test_outbound_tables_have_keys_and_constraints(client: AsyncClient):
    meta = await _inspect(
        lambda i: {
            "unique": i.get_unique_constraints("outbound_order"),
            "fks_order": i.get_foreign_keys("outbound_order"),
            "fks_item": i.get_foreign_keys("outbound_order_item"),
            "checks": i.get_check_constraints("outbound_order_item"),
            "indexes_order": i.get_indexes("outbound_order"),
            "indexes_item": i.get_indexes("outbound_order_item"),
        }
    )

    assert ["invoice_number"] in [u["column_names"] for u in meta["unique"]]

    def fk_map(fks):
        return {
            fk["constrained_columns"][0]: (
                fk["referred_table"],
                (fk.get("options") or {}).get("ondelete"),
            )
            for fk in fks
        }

    assert fk_map(meta["fks_order"]) == {
        "recipient_id": ("recipient", "RESTRICT"),
        "warehouse_id": ("warehouse", "RESTRICT"),
        "created_by_user_id": ("user", "RESTRICT"),
    }
    assert fk_map(meta["fks_item"]) == {
        "outbound_order_id": ("outbound_order", "CASCADE"),
        "product_id": ("product", "RESTRICT"),
    }
    assert "check_outbound_order_item_quantity_positive" in {c["name"] for c in meta["checks"]}

    indexed_order = {tuple(ix["column_names"]) for ix in meta["indexes_order"]}
    indexed_item = {tuple(ix["column_names"]) for ix in meta["indexes_item"]}
    assert {("recipient_id",), ("warehouse_id",)} <= indexed_order
    assert {("outbound_order_id",), ("product_id",)} <= indexed_item

# ---------------------------------------------------------------------------
# Обмеження на рівні БД
# ---------------------------------------------------------------------------

async def test_order_defaults_are_applied_by_database(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)

    order_id = await _insert_order(fixture)

    async with async_session_maker() as session:
        order = await session.get(OutboundOrder, order_id)
        assert order.order_status == OrderStatus.DRAFT
        assert order.created_at is not None
    assert await _count(OutboundOrderItem, OutboundOrderItem.outbound_order_id == order_id) == 0

@pytest.mark.parametrize("quantity", ["0", "-1", "-0.01"])
async def test_item_quantity_must_be_positive(client: AsyncClient, quantity: str):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    order_id = await _insert_order(fixture)

    with pytest.raises(IntegrityError):
        await _insert_item(order_id, fixture["product_id"], quantity)

    assert await _count(OutboundOrderItem, OutboundOrderItem.outbound_order_id == order_id) == 0

async def test_item_quantity_accepts_minimal_positive_value(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    order_id = await _insert_order(fixture)

    await _insert_item(order_id, fixture["product_id"], "0.01")

    async with async_session_maker() as session:
        quantity = await session.scalar(
            select(OutboundOrderItem.quantity).where(OutboundOrderItem.outbound_order_id == order_id)
        )
    assert quantity == Decimal("0.01")

async def test_invoice_number_must_be_unique(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    number = unique("UNIQ")
    await _insert_order(fixture, invoice_number=number)

    with pytest.raises(IntegrityError):
        await _insert_order(fixture, invoice_number=number)

async def test_order_requires_existing_references(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)

    for field in ("recipient_id", "warehouse_id", "created_by_user_id"):
        with pytest.raises(IntegrityError):
            await _insert_order(fixture, **{field: 999999})

async def test_item_requires_existing_order_and_product(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    order_id = await _insert_order(fixture)

    with pytest.raises(IntegrityError):
        await _insert_item(999999, fixture["product_id"], 1)
    with pytest.raises(IntegrityError):
        await _insert_item(order_id, 999999, 1)

async def test_deleting_order_cascades_to_items(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    order_id = await _insert_order(fixture)
    await _insert_item(order_id, fixture["product_id"], 1)
    await _insert_item(order_id, fixture["product_id"], 2, "pallet")

    async with async_session_maker() as session:
        await session.execute(delete(OutboundOrder).where(OutboundOrder.id == order_id))
        await session.commit()

    assert await _count(OutboundOrderItem, OutboundOrderItem.outbound_order_id == order_id) == 0

async def test_referenced_recipient_product_and_warehouse_cannot_be_deleted(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    order_id = await _insert_order(fixture)
    await _insert_item(order_id, fixture["product_id"], 1)

    for model, key in (
        (Recipient, fixture["recipient_id"]),
        (Product, fixture["product_id"]),
        (Warehouse, fixture["warehouse_id"]),
    ):
        async with async_session_maker() as session:
            with pytest.raises(IntegrityError):
                await session.execute(delete(model).where(model.id == key))
                await session.commit()

    assert await _count(OutboundOrder, OutboundOrder.id == order_id) == 1

async def test_api_delete_removes_items_from_database(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    order = await create_outbound(
        client, token, fixture, [line(fixture["product_id"], 1), line(fixture["product_id"], 2, "pallet")]
    )
    order_id = order["outbound_order_id"]
    assert await _count(OutboundOrderItem, OutboundOrderItem.outbound_order_id == order_id) == 2

    from tests.invoice_utils import OUTBOUND
    from tests.utils import auth

    assert (await client.delete(f"{OUTBOUND}/{order_id}", headers=auth(token))).status_code == 204

    assert await _count(OutboundOrderItem, OutboundOrderItem.outbound_order_id == order_id) == 0

async def test_stock_tables_enforce_quantity_rules(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    base = {
        "product_id": fixture["product_id"],
        "shelf_id": fixture["shelf_ids"][0],
        "warehouse_id": fixture["warehouse_id"],
    }

    for quantity in (Decimal("0"), Decimal("-1")):
        async with async_session_maker() as session:
            session.add(StockItem(**base, quantity=quantity, package_type="box"))
            with pytest.raises(IntegrityError):
                await session.commit()

    async with async_session_maker() as session:
        session.add(
            StockMovement(**base, quantity=Decimal("0"), movement_type=StockMovementType.MANUAL)
        )
        with pytest.raises(IntegrityError):
            await session.commit()

    # Від'ємний рух (списання) дозволений, нульовий - ні.
    async with async_session_maker() as session:
        session.add(
            StockMovement(**base, quantity=Decimal("-5"), movement_type=StockMovementType.OUTBOUND_ORDER)
        )
        await session.commit()

async def test_stock_item_is_unique_per_shelf_product_and_package(client: AsyncClient):
    token = await login(client, ADMIN)
    fixture = await make_fixture(client, token)
    base = {
        "product_id": fixture["product_id"],
        "shelf_id": fixture["shelf_ids"][0],
        "warehouse_id": fixture["warehouse_id"],
        "quantity": Decimal("1"),
    }
    async with async_session_maker() as session:
        session.add(StockItem(**base, package_type="box"))
        await session.commit()

    async with async_session_maker() as session:
        session.add(StockItem(**base, package_type="pallet"))  # інший тип упаковки - можна
        await session.commit()

    async with async_session_maker() as session:
        session.add(StockItem(**base, package_type="box"))
        with pytest.raises(IntegrityError):
            await session.commit()

# ---------------------------------------------------------------------------
# Міграції alembic (окрема тимчасова база)
# ---------------------------------------------------------------------------

def _scratch_url() -> str:
    return make_url(settings.DATABASE_URL).set(database=SCRATCH_DB).render_as_string(hide_password=False)

async def _admin_sql(statement: str) -> None:
    async with engine.connect() as connection:
        connection = await connection.execution_options(isolation_level="AUTOCOMMIT")
        await connection.execute(text(statement))

async def _alembic(*args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "DB_NAME": SCRATCH_DB}
    return await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-m", "alembic", *args],
        cwd=BACKEND_DIR,
        env=env,
        capture_output=True,
        text=True,
    )

async def _upgrade(target: str) -> None:
    result = await _alembic("upgrade", target)
    assert result.returncode == 0, result.stderr[-2000:]

@pytest_asyncio.fixture(loop_scope="session")
async def scratch_db():
    """Порожня база для прогону міграцій; пропускає тест, якщо створити її неможливо."""
    try:
        await _admin_sql(f'DROP DATABASE IF EXISTS "{SCRATCH_DB}" WITH (FORCE)')
        await _admin_sql(f'CREATE DATABASE "{SCRATCH_DB}"')
    except Exception as error:  # noqa: BLE001 - немає прав на CREATE DATABASE
        pytest.skip(f"Не вдалося створити тимчасову базу: {error}")

    scratch_engine = create_async_engine(_scratch_url())
    yield scratch_engine
    await scratch_engine.dispose()
    await _admin_sql(f'DROP DATABASE IF EXISTS "{SCRATCH_DB}" WITH (FORCE)')

def test_alembic_history_has_single_head():
    script = ScriptDirectory.from_config(Config(str(BACKEND_DIR / "alembic.ini")))

    heads = script.get_heads()

    assert len(heads) == 1, f"Кілька голів міграцій: {heads}"

def test_outbound_migration_is_part_of_history():
    script = ScriptDirectory.from_config(Config(str(BACKEND_DIR / "alembic.ini")))

    revisions = {revision.revision: revision for revision in script.walk_revisions()}

    assert "5a6b7c8d9e0f" in revisions
    assert "Create outbound orders" in revisions["5a6b7c8d9e0f"].doc

async def test_migrations_create_schema_matching_models(scratch_db):
    await _upgrade("head")

    async with scratch_db.connect() as connection:
        tables = await connection.run_sync(lambda sync: set(inspect(sync).get_table_names()))
        diffs = await connection.run_sync(
            lambda sync: compare_metadata(MigrationContext.configure(sync), Base.metadata)
        )

    assert INVOICE_TABLES <= tables
    relevant = [diff for diff in diffs if any(name in str(diff) for name in INVOICE_TABLES)]
    assert relevant == [], f"Міграції розходяться з моделями: {relevant}"

async def test_migrations_can_be_downgraded_and_upgraded_again(scratch_db):
    await _upgrade("head")

    down = await _alembic("downgrade", "-1")
    assert down.returncode == 0, down.stderr[-2000:]
    async with scratch_db.connect() as connection:
        tables = await connection.run_sync(lambda sync: set(inspect(sync).get_table_names()))
        columns = await connection.run_sync(
            lambda sync: {c["name"] for c in inspect(sync).get_columns("outbound_order")}
        )
    assert "stock_item" not in tables
    assert "warehouse_id" not in columns

    await _upgrade("head")
    async with scratch_db.connect() as connection:
        tables = await connection.run_sync(lambda sync: set(inspect(sync).get_table_names()))
    assert INVOICE_TABLES <= tables

async def test_migration_backfills_warehouse_for_existing_orders(scratch_db):
    """Накладні, створені до появи складів, отримують найменший існуючий склад."""
    await _upgrade(PREVIOUS_REVISION)
    async with scratch_db.begin() as connection:
        await connection.execute(text("INSERT INTO role (role_name) VALUES ('ADMINISTRATOR')"))
        await connection.execute(
            text(
                'INSERT INTO "user" (user_email, password_hash, first_name, last_name, role_id) '
                "VALUES ('legacy@example.com', 'x', 'A', 'B', (SELECT MIN(role_id) FROM role))"
            )
        )
        await connection.execute(
            text("INSERT INTO recipient (company_name, delivery_address) VALUES ('Legacy', 'Київ')")
        )
        await connection.execute(
            text(
                "INSERT INTO warehouse (warehouse_name, total_warehouse_area) "
                "VALUES ('W1', 10), ('W2', 20)"
            )
        )
        await connection.execute(text("INSERT INTO category (category_name) VALUES ('C')"))
        await connection.execute(
            text(
                "INSERT INTO product (sku_code, product_name, category_id, unit_of_measure, "
                "product_width, product_length, product_height, product_weight) "
                "VALUES ('SKU-LEGACY', 'P', (SELECT MIN(category_id) FROM category), 'PCS', 1, 1, 1, 1)"
            )
        )
        await connection.execute(
            text(
                "INSERT INTO outbound_order (recipient_id, invoice_number, planned_shipment_date, "
                "order_status, created_by_user_id) VALUES "
                "((SELECT MIN(recipient_id) FROM recipient), 'LEGACY-1', now(), 'DRAFT', "
                '(SELECT MIN(user_id) FROM "user"))'
            )
        )
        await connection.execute(
            text(
                "INSERT INTO outbound_order_item (outbound_order_id, product_id, quantity, package_type) "
                "VALUES ((SELECT MIN(outbound_order_id) FROM outbound_order), "
                "(SELECT MIN(product_id) FROM product), 3.5, 'box')"
            )
        )

    await _upgrade("head")

    async with scratch_db.connect() as connection:
        row = (
            await connection.execute(
                text(
                    "SELECT o.warehouse_id, (SELECT MIN(warehouse_id) FROM warehouse), "
                    "i.quantity, o.order_status "
                    "FROM outbound_order o JOIN outbound_order_item i "
                    "ON i.outbound_order_id = o.outbound_order_id "
                    "WHERE o.invoice_number = 'LEGACY-1'"
                )
            )
        ).one()
    assert row[0] == row[1]
    assert row[2] == Decimal("3.50")
    assert row[3] == "DRAFT"
