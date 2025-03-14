"""RAG服务测试模块

这个模块包含了RAG服务的单元测试。
"""

import unittest
from unittest.mock import MagicMock, patch

from app.services.rag.service import RAGService
from app.utils.exceptions import ServiceError

class TestRAGService(unittest.TestCase):
    """RAG服务测试类"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 创建模拟对象
        self.document_processor = MagicMock()
        self.llm_client = MagicMock()
        self.llm_response_handler = MagicMock()
        self.search_client = MagicMock()
        self.embedding_client = MagicMock()
        
        # 创建RAG服务实例
        self.rag_service = RAGService(
            document_processor=self.document_processor,
            llm_client=self.llm_client,
            llm_response_handler=self.llm_response_handler,
            search_client=self.search_client,
            embedding_client=self.embedding_client
        )
    
    def test_create_prompt(self):
        """测试创建提示"""
        # 准备测试数据
        query = "测试查询"
        search_results = "测试搜索结果"
        
        # 调用被测试的方法
        messages = self.rag_service.create_prompt(query, search_results)
        
        # 验证结果
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn(query, messages[1]["content"])
        self.assertIn(search_results, messages[1]["content"])
    
    def test_search_with_vector_search(self):
        """测试向量搜索"""
        # 准备测试数据
        query = "测试查询"
        query_embedding = [0.1, 0.2, 0.3]
        search_results = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "content": "测试内容1",
                            "metadata": {"file_name": "测试文件1"}
                        },
                        "_score": 0.9
                    },
                    {
                        "_source": {
                            "content": "测试内容2",
                            "metadata": {"file_name": "测试文件2"}
                        },
                        "_score": 0.8
                    }
                ]
            }
        }
        
        # 配置模拟对象
        self.embedding_client.get_embedding.return_value = [query_embedding]
        self.search_client.search.return_value = search_results
        self.rag_service.use_vector_search = True
        
        # 调用被测试的方法
        result = self.rag_service.search(query)
        
        # 验证结果
        self.embedding_client.get_embedding.assert_called_once_with([query])
        self.search_client.search.assert_called_once_with(query_vector=query_embedding, top_k=5)
        self.assertIn("测试内容1", result)
        self.assertIn("测试内容2", result)
        self.assertIn("测试文件1", result)
        self.assertIn("测试文件2", result)
    
    def test_search_with_simple_search(self):
        """测试简单搜索"""
        # 准备测试数据
        query = "测试查询"
        document_content = "测试文档内容"
        
        # 配置模拟对象
        self.rag_service.use_vector_search = False
        self.document_processor.read_all_documents.return_value = document_content
        
        # 调用被测试的方法
        result = self.rag_service.search(query)
        
        # 验证结果
        self.document_processor.read_all_documents.assert_called_once()
        self.assertEqual(result, document_content)
    
    def test_process(self):
        """测试处理用户查询"""
        # 准备测试数据
        query = "测试查询"
        search_results = "测试搜索结果"
        llm_response = MagicMock()
        response_json = {"choices": [{"message": {"content": "测试回复"}}]}
        
        # 配置模拟对象
        self.rag_service.search = MagicMock(return_value=search_results)
        self.llm_client.generate_response.return_value = llm_response
        self.llm_response_handler.handle_normal_response.return_value = response_json
        self.llm_response_handler.extract_content_from_response.return_value = "测试回复"
        
        # 调用被测试的方法
        result = self.rag_service.process(query)
        
        # 验证结果
        self.rag_service.search.assert_called_once_with(query)
        self.llm_client.generate_response.assert_called_once()
        self.llm_response_handler.handle_normal_response.assert_called_once_with(llm_response)
        self.llm_response_handler.extract_content_from_response.assert_called_once_with(response_json)
        self.assertEqual(result, "测试回复")
    
    def test_process_stream(self):
        """测试流式处理用户查询"""
        # 准备测试数据
        query = "测试查询"
        search_results = "测试搜索结果"
        llm_response = MagicMock()
        stream_chunks = [{"content": "测试"}, {"content": "回复"}]
        
        # 配置模拟对象
        self.rag_service.search = MagicMock(return_value=search_results)
        self.llm_client.generate_response.return_value = llm_response
        self.llm_response_handler.handle_stream_response.return_value = stream_chunks
        
        # 调用被测试的方法
        result = list(self.rag_service.process_stream(query))
        
        # 验证结果
        self.rag_service.search.assert_called_once_with(query)
        self.llm_client.generate_response.assert_called_once()
        self.llm_response_handler.handle_stream_response.assert_called_once_with(llm_response)
        self.assertEqual(result, stream_chunks)
    
    def test_index_documents(self):
        """测试索引文档"""
        # 准备测试数据
        documents = [
            {"content": "测试内容1", "metadata": {"file_name": "测试文件1"}},
            {"content": "测试内容2", "metadata": {"file_name": "测试文件2"}}
        ]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        # 配置模拟对象
        self.rag_service.use_vector_search = True
        self.embedding_client.get_embedding.side_effect = [[e] for e in embeddings]
        self.search_client.bulk_index_documents.return_value = {"success": 2, "errors": 0}
        
        # 调用被测试的方法
        result = self.rag_service.index_documents(documents)
        
        # 验证结果
        self.assertEqual(self.embedding_client.get_embedding.call_count, 2)
        self.search_client.bulk_index_documents.assert_called_once()
        self.assertEqual(result["success"], 2)
        self.assertEqual(result["errors"], 0)


if __name__ == '__main__':
    unittest.main() 