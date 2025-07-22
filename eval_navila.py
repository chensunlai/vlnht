import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import cv2
import json
import os
import argparse
import queue
import requests
import habitat_extensions
from utils.benchmark import Benchmark

SERVER_URL = "http://10.82.1.223:18880"


class NaVILAAgent(habitat.Agent):
    _env = None
    task_id = None
    actque = queue.Queue()

    def reset(self):
        self.actque.queue.clear()
        self.task_id = self.create_task()

        self.task_name = "{}-{}".format(self._env.current_episode.scene_id.split(
            "/")[-2], self._env.current_episode.episode_id)
        self.log_path = os.path.join("runs", self.task_name, "log.jsonl")
        self.result_path = os.path.join("runs", self.task_name, "result.json")
        self.img_path = os.path.join("runs", self.task_name, "images")
        self.realtime_image_path = os.path.join("runs", "realtime.jpg")
        if not os.path.exists(os.path.join("runs", self.task_name)):
            os.makedirs(self.img_path)

    def create_task(self) -> str:
        r = requests.post(f"{SERVER_URL}/api/tasks", timeout=10)
        r.raise_for_status()
        return r.json()["task_id"]

    def upload_image(self, task_id: str, data: bytes) -> int:
        files = {"file": ("frame.jpg", data, "image/jpeg")}
        r = requests.post(
            f"{SERVER_URL}/api/tasks/{task_id}/images", files=files, timeout=10)
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

        img = cv2.cvtColor(observations["rgb"], cv2.COLOR_RGB2BGR)
        ret, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        img_bytes = buf.tobytes()
        image_idx = self.upload_image(self.task_id, img_bytes)
        text = self.infer(self.task_id, observations['instruction']['text'])

        result = {
            "image": f"image/{image_idx}.jpg",
            "instruct": observations['instruction']['text'],
            "response": text,
            "goal": self._env.current_episode.goals[0].position,
            "now": self._env.sim.get_agent_state().position.tolist(),
            "metrics": self._env.get_metrics()
        }

        cv2.imwrite(os.path.join(self.img_path, f"{image_idx}.jpg"), img)
        cv2.imwrite(self.realtime_image_path, img)

        with open(self.log_path, "a+") as fp:
            fp.write(json.dumps(result, ensure_ascii=False) + "\n")

        for _ in range(4):
            try:
                text = text.lower()
                if "left" in text:
                    [self.actque.put({"action": HabitatSimActions.turn_left})
                    for i in range(round(float(text.split()[-2])/15))]
                elif "right" in text:
                    [self.actque.put({"action": HabitatSimActions.turn_right})
                    for i in range(round(float(text.split()[-2])/15))]
                elif "forward" in text:
                    [self.actque.put({"action": HabitatSimActions.move_forward})
                    for i in range(round(float(text.split()[-2])/25))]

                if not self.actque.empty():
                    return self.actque.get()
                break
            except:
                self.actque.queue.clear()
        
        return {"action": HabitatSimActions.stop}

    def EpisodeOverCallback(self, metrics):
        with open(self.result_path, "a+") as fp:
            fp.write(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-config",
        type=str,
        default="config/navila_r2r.yaml",
    )
    args = parser.parse_args()

    agent = NaVILAAgent()
    benchmark = Benchmark(args.task_config)
    agent._env = benchmark._env
    metrics = benchmark.evaluate(agent)

    for k, v in metrics.items():
        print("{}: {:.3f}".format(k, v))


if __name__ == "__main__":
    main()
