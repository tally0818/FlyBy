'Internal trajectory cleanup tool.'

from .base import BaseTool, register_tool


@register_tool
class FinishTool(BaseTool):
    tool_type = "finish"

    def __init__(self, num_workers=1, other_tools=None, **kwargs):
        super().__init__(num_workers=num_workers)
        self.other_tools = other_tools or {}

    def get_action_priority(self, action: str, extra_field: dict) -> int:

        return -1

    def conduct_action(self, trajectory_id, action, extra_field):
        for tool in self.other_tools.values():
            delete_env = getattr(tool, "delete_env", None)
            if delete_env is None:
                continue
            remote_delete = getattr(delete_env, "remote", None)
            if remote_delete is not None:
                remote_delete(trajectory_id)
            else:
                delete_env(trajectory_id)

        return "", True, True
