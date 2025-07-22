from habitat.core.embodied_task import SimulatorTaskAction
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions

@registry.register_task_action()
class VLNStopAction(SimulatorTaskAction):
    name = "STOP"

    def step(self, *args, **kwargs):
        return self._sim.step(HabitatSimActions.stop)

@registry.register_task_action()
class VLNForwardAction(SimulatorTaskAction):
    name = "MOVE_FORWARD_VAR"

    def step(self, *args, **kwargs):
        distance = float(kwargs.get("amount", 0.25))   
        agent = self._sim.get_agent(0)           
        act_space = agent.action_space[HabitatSimActions.move_forward.name]
        spec = act_space.actuation              
        old_amount = spec.amount
        spec.amount = distance
        obs = self._sim.step(HabitatSimActions.move_forward)
        spec.amount = old_amount
        return obs
    
@registry.register_task_action()
class VLNTurnLeftAction(SimulatorTaskAction):
    name = "TURN_LEFT_VAR"

    def step(self, *args, **kwargs):
        angle = kwargs.get("amount", 15)   
        return self._sim.step(
            HabitatSimActions.turn_left,
            {"amount": float(angle)},
        )

@registry.register_task_action()
class VLNTurnRightAction(SimulatorTaskAction):
    name = "TURN_RIGHT_VAR"

    def step(self, *args, **kwargs):
        angle = kwargs.get("amount", 15)   
        return self._sim.step(
            HabitatSimActions.turn_right,
            {"amount": float(angle)},
        )

@registry.register_task_action()
class VLNLookUpAction(SimulatorTaskAction):
    name = "LOOK_UP_VAR"

    def step(self, *args, **kwargs):
        angle = kwargs.get("amount", 15)   
        return self._sim.step(
            HabitatSimActions.look_up,
            {"amount": float(angle)},
        )

@registry.register_task_action()
class VLNLookDownAction(SimulatorTaskAction):
    name = "LOOK_DOWN_VAR"

    def step(self, *args, **kwargs):
        angle = kwargs.get("amount", 15)   
        return self._sim.step(
            HabitatSimActions.look_down,
            {"amount": float(angle)},
        )