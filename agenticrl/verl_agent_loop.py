"""A verl tool loop that stops immediately after submit_solution."""

from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop


class BirdSqlAgentLoop(ToolAgentLoop):
    async def _handle_processing_tools_state(self, agent_data):
        state = await super()._handle_processing_tools_state(agent_data)
        if agent_data.extra_fields.get("bird_submitted"):
            return AgentState.TERMINATED
        return state
