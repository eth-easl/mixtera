from mixtera.network.server_task import ServerTask

import unittest

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

    def test_unique_values(self):
        values = set(member.value for member in ServerTask)
        self.assertEqual(len(values), len(ServerTask))

    def test_invalid_value(self):
        with self.assertRaises(ValueError):
            ServerTask(5)

if __name__ == '__main__':
    unittest.main()