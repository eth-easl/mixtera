from mixtera.datasets import MixteraDataset


class RemoteMixteraDataset(MixteraDataset):

    def __init__(self, endpoint: str) -> None:
        # Idea: Server holds a LocalDataset, which we can interact with via the RemoteDataset.
        raise NotImplementedError("RemoteMixteraDataset has not been implemented yet.")
