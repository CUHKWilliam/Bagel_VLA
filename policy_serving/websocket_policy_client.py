import asyncio
import http
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)

class WebsocketPolicyClient(_base_policy.BasePolicy):
    def __init__(self, host: str = "localhost", port: Optional[int] = 8080, api_key: Optional[str] = None) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()
        
        # 记录服务器元数据中的VLA策略信息
        if "vla_policy_info" in self._server_metadata:
            logging.info(f"Connected to VLA policy server: {self._server_metadata['vla_policy_info']}")

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for VLA policy server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                logging.info(f"Connected to VLA policy server")
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for VLA policy server...")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:
        # VLA策略特定的obs格式转换
        vla_obs = self._convert_to_vla_format(obs)
        
        data = self._packer.pack(vla_obs)
        self._ws.send(data)
        response = self._ws.recv()
        
        if isinstance(response, str):
            raise RuntimeError(f"Error in VLA inference server:\n{response}")
        
        result = msgpack_numpy.unpackb(response)
        return result
    
    def _convert_to_vla_format(self, obs: Dict) -> Dict:
        """将通用obs格式转换为VLA策略需要的格式"""
        vla_obs = {
            "robot0_eye_in_hand_image": obs.get("wrist_image"),
            "agentview_image": obs.get("main_image"),
            "robot0_eef_pos": obs.get("eef_position", np.zeros(3)),
            "robot0_eef_quat": obs.get("eef_orientation", np.array([1, 0, 0, 0])),
            "robot0_gripper_qpos": obs.get("gripper_position", np.zeros(1)),
            "task_description": obs.get("task", "default task"),
        }
        return vla_obs

    @override
    def reset(self) -> None:
        """发送重置信号到服务器"""
        reset_msg = {"reset": True}
        data = self._packer.pack(reset_msg)
        self._ws.send(data)
        # 接收确认
        response = self._ws.recv()
        logging.info("VLA policy reset requested")
        
# client_example.py
import numpy as np

if __name__ == "__main__":
    
    client = WebsocketPolicyClient(host="localhost", port=8080)

    obs = {
        "robot0_eye_in_hand_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "agentview_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "robot0_eef_pos": np.array([0.5, 0.2, 0.3]),
        "robot0_eef_quat": np.array([0.707, 0, 0.707, 0]),  # w, x, y, z
        "robot0_gripper_qpos": np.array([0.1]),
        "task_description": "pick up the red block",
    }

    result = client.infer(obs)
    print(f"Action: {result['action']}")
    print(f"Inference time: {result['server_timing']['vla_infer_ms']} ms")

    client.reset()