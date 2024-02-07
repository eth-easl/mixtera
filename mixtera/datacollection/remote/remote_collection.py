from mixtera.datacollection import MixteraDataCollection


class RemoteDataCollection(MixteraDataCollection):

    def __init__(self, endpoint: str) -> None:
        # Idea: Server holds a LocalDataCollection, which we can interact with via the RemoteDataCollection
        raise NotImplementedError("RemoteDataCollection has not been implemented yet.")
