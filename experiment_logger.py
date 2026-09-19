import os
import time
import pandas as pd


class ExperimentLogger:
    def __init__(self, filename="route_separation_reuse1.csv"):
        self.filename = filename
        self.start_times = {}

        # Create CSV with header if it doesn't exist
        if not os.path.exists(self.filename):
            pd.DataFrame(columns=[
                "iteration",
                "pattern",
                "agent_1",
                "agent_2",
                "tree_id",
                "event_type",
                "usage_count",
                "agents_knowing_ccsm",
                "known_agents",
                "reused_from",
                "regret_time",
                "lookup_time",
                "lcp_time",
                "csm_time",
                "synthesis_time"
            ]).to_csv(self.filename, index=False)

    def start_timer(self, name):
        self.start_times[name] = time.perf_counter()

    def stop_timer(self, name):
        if name not in self.start_times:
            return 0.0
        elapsed = time.perf_counter() - self.start_times[name]
        del self.start_times[name]
        return elapsed

    def count_agents_knowing_tree(self, all_agents, tree_id):
        return sum(
            1 for ag in all_agents.values()
            if tree_id in ag.contour_trees
        )

    def known_agents_for_tree(self, all_agents, tree_id):
        return sorted([
            ag_id
            for ag_id, ag in all_agents.items()
            if tree_id in ag.contour_trees
        ])

    def log_event(
        self,
        iteration,
        pattern,
        agent_1,
        agent_2,
        tree_id,
        event_type,
        all_agents,
        usage_count=0,
        reused_from=None,
        lcp_time=0.0,
        csm_time=0.0,
        regret_time=0.0,
        lookup_time=0.0
    ):

        agents_knowing = self.count_agents_knowing_tree(all_agents, tree_id)
        known_agents = self.known_agents_for_tree(all_agents, tree_id)

        row = {
            "iteration": iteration,
            "pattern": pattern,
            "agent_1": agent_1,
            "agent_2": agent_2,
            "tree_id": tree_id,
            "event_type": event_type,
            "usage_count": usage_count,
            "agents_knowing_ccsm": agents_knowing,
            "known_agents": ";".join(map(str, known_agents)),
            "reused_from": reused_from,
            "regret_time": regret_time,
            "lookup_time": lookup_time,
            "lcp_time": lcp_time,
            "csm_time": csm_time,
            "synthesis_time": lcp_time + csm_time
        }

        # Immediately append one row to the CSV
        pd.DataFrame([row]).to_csv(
            self.filename,
            mode="a",
            header=False,
            index=False
        )

    def save(self,filename):
        # Nothing to do—everything is already on disk.
        print(f"Results already saved to {self.filename}")


EXP_LOGGER = ExperimentLogger()

