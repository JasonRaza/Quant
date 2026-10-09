"""REST API for the paper trading account.

Run: quant-paper-api  (or: uvicorn paper_trading.api:create_app --factory --reload)
Docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from paper_trading.broker import PaperBroker
from paper_trading.service import ALLOCATION_METHODS, Settings, make_broker, target_weights


class OrderRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=10, examples=["AAPL"])
    side: Literal["buy", "sell"]
    quantity: int = Field(..., gt=0)


class RebalanceRequest(BaseModel):
    method: str = Field("risk_parity", examples=ALLOCATION_METHODS)
    max_weight: float = Field(0.25, gt=0, le=1)


class ResetRequest(BaseModel):
    initial_cash: float = Field(100_000.0, gt=0)


def create_app(settings: Optional[Settings] = None, broker: Optional[PaperBroker] = None):
    settings = settings or Settings.from_config()
    broker = broker or make_broker(settings)
    app = FastAPI(title="Quant Paper Trading", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/account")
    def account():
        return broker.account()

    @app.get("/positions")
    def positions():
        return broker.positions().round(4).to_dict(orient="records")

    @app.get("/orders")
    def orders(limit: int = Query(100, ge=1, le=1000)):
        df = broker.orders(limit)
        return df.astype(object).where(df.notna(), None).to_dict(orient="records")

    @app.post("/orders")
    def submit_order(req: OrderRequest):
        order = broker.submit_order(req.symbol, req.side, req.quantity)
        if order.status != "filled":
            raise HTTPException(status_code=400, detail=order.to_dict())
        return order.to_dict()

    @app.get("/prices/{symbol}")
    def price(symbol: str):
        px = broker.price_source(symbol.upper())
        if px is None:
            raise HTTPException(status_code=404, detail=f"no data for {symbol.upper()}")
        return {"symbol": symbol.upper(), "price": px}

    @app.get("/allocations/{method}")
    def allocation(method: str, max_weight: float = Query(0.25, gt=0, le=1)):
        weights = _weights_or_400(settings, method, max_weight)
        return {"method": method, "weights": weights.round(4).to_dict()}

    @app.post("/rebalance")
    def rebalance(req: RebalanceRequest):
        weights = _weights_or_400(settings, req.method, req.max_weight)
        orders = broker.rebalance(weights.to_dict())
        return {
            "method": req.method,
            "orders": [o.to_dict() for o in orders],
            "account": broker.account(),
        }

    @app.post("/reset")
    def reset(req: ResetRequest):
        broker.reset(req.initial_cash)
        return broker.account()

    return app


def _weights_or_400(settings: Settings, method: str, max_weight: float):
    try:
        return target_weights(settings, method, max_weight)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def main() -> None:
    import uvicorn

    uvicorn.run("paper_trading.api:create_app", factory=True, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
