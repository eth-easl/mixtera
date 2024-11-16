import unittest

from mixtera.network.server_task import ServerTask


class TestServerTask(unittest.TestCase):
    def test_register_query(self):
        self.assertEqual(ServerTask.REGISTER_QUERY, 0)
        self.assertEqual(ServerTask(0), ServerTask.REGISTER_QUERY)

    def test_read_file(self):
        self.assertEqual(ServerTask.READ_FILE, 1)
        self.assertEqual(ServerTask(1), ServerTask.READ_FILE)

    def test_get_query_id(self):
        self.assertEqual(ServerTask.GET_QUERY_ID, 2)
        self.assertEqual(ServerTask(2), ServerTask.GET_QUERY_ID)

    def test_get_meta_result(self):
        self.assertEqual(ServerTask.GET_META_RESULT, 3)
        self.assertEqual(ServerTask(3), ServerTask.GET_META_RESULT)

    def test_get_next_result_chunk(self):
        self.assertEqual(ServerTask.GET_NEXT_RESULT_CHUNK, 4)
        self.assertEqual(ServerTask(4), ServerTask.GET_NEXT_RESULT_CHUNK)

    def test_register_dataset(self):
        self.assertEqual(ServerTask.REGISTER_DATASET, 5)
        self.assertEqual(ServerTask(5), ServerTask.REGISTER_DATASET)

    def test_register_metadata_parser(self):
        self.assertEqual(ServerTask.REGISTER_METADATA_PARSER, 6)
        self.assertEqual(ServerTask(6), ServerTask.REGISTER_METADATA_PARSER)

    def test_check_dataset_exists(self):
        self.assertEqual(ServerTask.CHECK_DATASET_EXISTS, 7)
        self.assertEqual(ServerTask(7), ServerTask.CHECK_DATASET_EXISTS)

    def test_list_datasets(self):
        self.assertEqual(ServerTask.LIST_DATASETS, 8)
        self.assertEqual(ServerTask(8), ServerTask.LIST_DATASETS)

    def test_remove_dataset(self):
        self.assertEqual(ServerTask.REMOVE_DATASET, 9)
        self.assertEqual(ServerTask(9), ServerTask.REMOVE_DATASET)

    def test_add_property(self):
        self.assertEqual(ServerTask.ADD_PROPERTY, 10)
        self.assertEqual(ServerTask(10), ServerTask.ADD_PROPERTY)

    def test_receive_feedback(self):
        self.assertEqual(ServerTask.RECEIVE_FEEDBACK, 14)
        self.assertEqual(ServerTask(14), ServerTask.RECEIVE_FEEDBACK)

    def test_unique_values(self):
        values = set(member.value for member in ServerTask)
        self.assertEqual(len(values), len(ServerTask))

    def test_invalid_value(self):
        with self.assertRaises(ValueError):
            ServerTask(15)
