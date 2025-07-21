import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import cv2
import matplotlib.pyplot as plt
from habitat.config.default import get_agent_config
from habitat.core.registry import registry
from habitat.core.embodied_task import (
    EmbodiedTask,
    Measure,
    SimulatorTaskAction,
)
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
import argparse
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import habitat_extensions
import queue
import requests

SERVER_URL = "http://10.82.1.223:18880"    

class NaVILAAgent(habitat.Agent):
    _env = None
    task_id = None
    actque = queue.Queue()

    def reset(self):
        self.actque.queue.clear()
        self.task_id = self.create_task()
        pass
    
    def create_task(self) -> str:
        r = requests.post(f"{SERVER_URL}/api/tasks", timeout=10)
        r.raise_for_status()
        return r.json()["task_id"]

    def upload_image(self, task_id: str, data: bytes) -> int:
        files = {"file": ("frame.jpg", data, "image/jpeg")}
        r = requests.post(f"{SERVER_URL}/api/tasks/{task_id}/images", files=files, timeout=10)
        r.raise_for_status()
        return r.json()["image_count"]

    def infer(self, task_id: str, instr: str) -> str:
        payload = {"instruction": instr}
        r = requests.post(
            f"{SERVER_URL}/api/tasks/{task_id}/infer",
            json=payload,
            timeout=60,        
        )
        r.raise_for_status()
        return r.json()["result"]

    def act(self, observations):
        if not self.actque.empty():
            return self.actque.get()

        ret, buf = cv2.imencode(".jpg", cv2.cvtColor(observations["rgb"], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        img_bytes = buf.tobytes()
        self.upload_image(self.task_id, img_bytes)
        text = self.infer(self.task_id, observations['instruction']['text'])
        
        # print(text)

        text = text.lower()
        if "left" in text:
            [self.actque.put({"action": HabitatSimActions.turn_left}) for i in range(round(float(text.split()[-2])/15))]
        elif "right" in text:
            [self.actque.put({"action": HabitatSimActions.turn_right}) for i in range(round(float(text.split()[-2])/15))]
        elif "forward" in text:
            [self.actque.put({"action": HabitatSimActions.move_forward}) for i in range(round(float(text.split()[-2])/25))]
            
        if not self.actque.empty():
            return self.actque.get()

        return {"action": HabitatSimActions.stop}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-config",
        type=str,
        default="config/navila_r2r.yaml",
    )
    args = parser.parse_args()

    agent = NaVILAAgent()
    benchmark = habitat.Benchmark(args.task_config)
    agent._env = benchmark._env
    metrics = benchmark.evaluate(agent, num_episodes=10)

    for k, v in metrics.items():
        print("{}: {:.3f}".format(k, v))


if __name__ == "__main__":
    main()
