from abc import ABC, abstractmethod
from typing import Dict, List, Optional

STATUS_READY = 0
STATUS_RUNNING = 1
STATUS_INTERCHANGE_DONE = 2
STATUS_GRAPH_READY = 3
STATUS_GRAPH_DONE = 4


class ExperimentManagerInterface(ABC):
    @abstractmethod
    def insert(self, opts: Dict) -> int: ...

    @abstractmethod
    def update(self, opts: Dict, id: int) -> None: ...

    @abstractmethod
    def fetch(self, n: Optional[int] = None, status: int = STATUS_READY) -> List[Dict]: ...

    @abstractmethod
    def query(
        self,
        cols: Optional[List[str]] = None,
        status: Optional[int] = None,
        abstraction: Optional[str] = None,
        id: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Dict]: ...


class ExperimentInterface(ABC):
    def __init__(self, finished_status: int = STATUS_INTERCHANGE_DONE):
        self.finished_status = finished_status

    @abstractmethod
    def experiment(self, opts: Dict) -> Dict: ...

    def run(self, opts: Dict, manager: ExperimentManagerInterface) -> Dict:
        res_dict = self.experiment(opts)
        res_dict["status"] = self.finished_status
        manager.update(res_dict, opts["id"])
        return res_dict
