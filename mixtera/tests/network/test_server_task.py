import unittest

from mixtera.network.server_task import ServerTask


class TestServerTask(unittest.TestCase):
    def test_register_query(self):
        self.assertEqual(ServerTask.REGISTER_QUERY, 0)
        self.assertEqual(ServerTask(0), ServerTask.REGISTER_QUERY)

    def test_query_exec_status(self):
        self.assertEqual(ServerTask.QUERY_EXEC_STATUS, 1)
        self.assertEqual(ServerTask(1), ServerTask.QUERY_EXEC_STATUS)

    def test_read_file(self):
        self.assertEqual(ServerTask.READ_FILE, 2)
        self.assertEqual(ServerTask(2), ServerTask.READ_FILE)

    def test_get_query_id(self):
        self.assertEqual(ServerTask.GET_QUERY_ID, 3)
        self.assertEqual(ServerTask(3), ServerTask.GET_QUERY_ID)

    def test_get_meta_result(self):
        self.assertEqual(ServerTask.GET_META_RESULT, 4)
        self.assertEqual(ServerTask(4), ServerTask.GET_META_RESULT)

    def test_get_next_result_chunk(self):
        self.assertEqual(ServerTask.GET_NEXT_RESULT_CHUNK, 5)
        self.assertEqual(ServerTask(5), ServerTask.GET_NEXT_RESULT_CHUNK)

    def test_register_dataset(self):
        self.assertEqual(ServerTask.REGISTER_DATASET, 6)
        self.assertEqual(ServerTask(6), ServerTask.REGISTER_DATASET)

    def test_dataset_status(self):
        self.assertEqual(ServerTask.DATASET_REGISTRATION_STATUS, 7)
        self.assertEqual(ServerTask(7), ServerTask.DATASET_REGISTRATION_STATUS)

    def test_register_metadata_parser(self):
        self.assertEqual(ServerTask.REGISTER_METADATA_PARSER, 8)
        self.assertEqual(ServerTask(8), ServerTask.REGISTER_METADATA_PARSER)

    def test_check_dataset_exists(self):
        self.assertEqual(ServerTask.CHECK_DATASET_EXISTS, 9)
        self.assertEqual(ServerTask(9), ServerTask.CHECK_DATASET_EXISTS)

    def test_list_datasets(self):
        self.assertEqual(ServerTask.LIST_DATASETS, 10)
        self.assertEqual(ServerTask(10), ServerTask.LIST_DATASETS)

    def test_remove_dataset(self):
        self.assertEqual(ServerTask.REMOVE_DATASET, 11)
        self.assertEqual(ServerTask(11), ServerTask.REMOVE_DATASET)

    def test_add_property(self):
        self.assertEqual(ServerTask.ADD_PROPERTY, 12)
        self.assertEqual(ServerTask(12), ServerTask.ADD_PROPERTY)

    def test_receive_feedback(self):
        self.assertEqual(ServerTask.RECEIVE_FEEDBACK, 16)
        self.assertEqual(ServerTask(16), ServerTask.RECEIVE_FEEDBACK)

    def test_unique_values(self):
        values = set(member.value for member in ServerTask)
        self.assertEqual(len(values), len(ServerTask))

    def test_invalid_value(self):
        with self.assertRaises(ValueError):
            ServerTask(17)
