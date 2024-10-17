class Cell2NetConfig:
    """Config manager for cell2net."""

    def __init__(
        self,
        batch_size: int = 128,
        dl_num_works: int = 4,
        random_state: int = 42,
    ) -> None:
        self.batch_size = batch_size
        self.dl_num_works = dl_num_works
        self.random_state = random_state


settings = Cell2NetConfig()
