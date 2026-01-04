import asyncio
import http
import logging
import time
import traceback
import argparse

import base_policy as _base_policy
import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames
from bagel_vla_policy import BagelVLAPolicy

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    def __init__(
        self,
        checkpoint_path: str,
        host: str = "0.0.0.0",
        port: int | None = None,
        device: str = "cuda",
        metadata: dict | None = None,
    ) -> None:
        self.bagel_vla = BagelVLAPolicy(checkpoint_path, device)
        
        self._host = host
        self._port = port
        
        self._metadata = metadata or {}
        self._metadata.update({
            "policy_type": "Bagel_VLA",
            "vla_policy_info": {
                "checkpoint_path": checkpoint_path,
                "device": device,
            }
        })
        
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            logger.info(f"VLA Policy Websocket Server started at ws://{self._host}:{self._port}")
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"VLA policy connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                
                obs_data = await websocket.recv()
                obs = msgpack_numpy.unpackb(obs_data)
                
                if isinstance(obs, dict) and obs.get("reset", False):
                    logger.info(f"VLA policy reset !!!")
                    self.bagel_vla.reset()
                    await websocket.send(packer.pack({"status": "reset_complete"}))
                    continue
                
                # VLA infer
                infer_time = time.monotonic()
                action = self.bagel_vla.infer(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "vla_infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000
                
                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"VLA policy connection from {websocket.remote_address} closed")
                break
            except Exception as e:
                logger.error(f"Error in VLA policy inference: {e}")
                error_msg = traceback.format_exc()
                await websocket.send(packer.pack({"error": error_msg}))
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="VLA policy inference error",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "VLA Policy Server OK\n")
    if request.path == "/vla_policy_info":
        return connection.respond(http.HTTPStatus.OK, "VLA Policy Info Endpoint\n")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8080)
    
    args = parser.parse_args()
    
    server = WebsocketPolicyServer(
        checkpoint_path=args.checkpoint,
        host=args.host,
        port=args.port,
    )
    
    print(f"Starting VLA Policy Websocket Server on ws://{args.host}:{args.port}")
    server.serve_forever()

if __name__ == "__main__":
    main()