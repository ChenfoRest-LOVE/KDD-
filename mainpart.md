# RAG 模型推理主程序说明文档

## 项目概述
本文档对应知识库中 **“Baseline pipeline实现”** 的核心代码，实现了 **Domain Router → Dynamic Router → Retriever → Reranker → Reader** 的端到端RAG（检索增强生成）流水线。代码支持批量数据处理、多模块灵活配置，并原生适配知识库中定义的实验流程与评估标准。


## 代码结构与核心模块

### 整体架构
```
├── 数据加载模块        # load_data_in_batches (批量读取JSONL/压缩文件)
├── 预测生成模块        # generate_predictions (调用RAG模型批量推理)
├── 组件初始化模块      # main函数（模型/检索器/路由器配置）
└── 结果保存模块        # 输出JSONL格式预测结果（含问题/标准答案/预测答案）
```


### 1. 数据加载模块（对应“数据准备与索引构建”）
**函数**：`load_data_in_batches(dataset_path, batch_size)`  
**功能**：从压缩文件（`.bz2`）或普通JSONL文件中读取数据，按`BATCH_SIZE=20`生成批次数据。  

#### 输入数据格式（JSONL每行结构）
```json
{
    "interaction_id": "样本唯一标识", 
    "query": "用户问题文本", 
    "search_results": ["候选chunk1", "候选chunk2"],  // 预检索的文档片段
    "query_time": "2024-01-01T12:00:00Z",  // ISO时间戳（供动态路由器判断时效性）
    "answer": "标准答案（用于评估）"
}
```

#### 关键特性
- 支持压缩文件读取（大规模数据集存储优化）
- 自动跳过JSON解码失败的行（提升鲁棒性）
- 生成器模式输出批次数据（降低内存占用）


### 2. 预测生成模块（对应“pipeline推理”）
**函数**：`generate_predictions(dataset_path, participant_model)`  
**功能**：调用RAG模型批量生成答案，分离`ground_truth`与`prediction`，并记录日志。  

#### 核心流程
1. 从批次数据中移除`answer`字段作为标准答案  
2. 调用`participant_model.batch_generate_answer(batch)`执行推理  
3. 记录样例日志（问题/标准答案/预测结果）以便调试  
4. 返回`queries`/`ground_truths`/`predictions`列表  


### 3. 组件初始化（对应“端到端Pipeline”）
在`if __name__ == "__main__":`中完成以下核心组件配置：

#### 3.1 生成模型（Reader模块）
```python
# 加载GPT-4o作为答案生成器（支持API或本地Ollama部署）
chat_model = load_model(
    model_name="gpt-4o", 
    api_key="<your-api-key>", 
    base_url="<your-base-url>", 
    temperature=0  # 确定性输出，避免随机生成
)
```
- **功能**：基于检索结果生成最终答案，对应知识库中 **“Reader/Summarizer”** 模块。


#### 3.2 检索器（Retriever + Reranker流程）
```python
# 初始化检索器（稠密检索+重排序）
retriever = Retriever(
    top_k_retrieve=10,  # 检索Top-10候选chunk
    top_k_rerank=5,    # 重排后保留Top-5
    embedding_model_path="models/retrieve/embedding_models/bge-m3",  # 嵌入模型
    reranker_model_path="models/retrieve/reranker_models/bge-reranker-v2-m3",  # 重排模型
    rerank=True        # 启用精排
)
```
- **功能**：对应知识库 **“Retriever + Reranker”** 流程，先通过`bge-m3`嵌入模型检索候选chunk，再用`bge-reranker-v2-m3`精排。  
- **扩展**：支持Milvus向量数据库（大规模数据场景），需取消相关注释并配置`collection_name`和`uri`。


#### 3.3 领域路由器（Domain Router模块）
```python
domain_router = SequenceClassificationRouter(
    model_path="models/router/bge-m3/domain",  # 领域分类模型路径
    classes=["finance", "music", "movie", "sports", "open"],  # 支持领域
    device_map="auto"  # 自动选择GPU/CPU
)
```
- **功能**：将问题分类到特定领域（如金融/体育），优化检索针对性，对应知识库 **“Domain Router”** 模块。


#### 3.4 动态路由器（Dynamic Router模块）
```python
dynamic_router = SequenceClassificationRouter(
    model_path="models/router/bge-m3/dynamic",  # 动态性分类模型路径
    classes=['static', 'slow-changing', 'fast-changing', 'real-time'],  # 动态性类别
    device_map="auto"
)
```
- **功能**：判定问题动态性（如实时事件/静态知识），决定是否触发实时API/KG访问，对应知识库 **“Dynamic Router”** 模块。


#### 3.5 RAG模型整合
```python
rag_model = RAGModel(
    chat_model=chat_model,          # 生成模型（Reader）
    retriever=retriever,            # 检索器
    domain_router=domain_router,    # 领域路由器
    dynamic_router=dynamic_router,  # 动态路由器
    use_kg=True                     # 启用知识图谱集成
)
```
- **功能**：整合所有组件，实现 **“问题→路由→检索→生成”** 的端到端流程，对应知识库 **“USTC基线Pipeline”**。


### 4. 推理与结果保存（对应“实验推理与结果记录”）
```python
# 批量推理
queries, ground_truths, predictions = generate_predictions(
    dataset_path="example_data/dev_data.jsonl.bz2",  # 输入数据集
    participant_model=rag_model                      # RAG模型实例
)

# 保存结果（JSONL格式，用于后续评估）
output_path = f"results/{model_name}_predictions.jsonl"
with open(output_path, "w", encoding="utf-8") as file:
    for query, ground_truth, prediction in zip(queries, ground_truths, predictions):
        item = {"query": query, "ground_truth": ground_truth, "prediction": prediction}
        file.write(json.dumps(item, ensure_ascii=False) + "\n")
```
- **输出格式**：每个样本包含`query`（问题）、`ground_truth`（标准答案）、`prediction`（模型预测），符合知识库中 **“实验记录与可复现性”** 要求。


## 环境配置与依赖

### 必要环境变量
```bash
# 动态路由器触发实时API调用（如KG查询）的模拟接口地址
export CRAG_MOCK_API_URL="http://localhost:8000"
```

### 模型依赖路径
| 组件           | 路径示例                                   |
|----------------|------------------------------------------|
| 嵌入模型       | `models/retrieve/embedding_models/bge-m3` |
| 重排模型       | `models/retrieve/reranker_models/bge-reranker-v2-m3` |
| 领域路由器模型 | `models/router/bge-m3/domain`           |
| 动态路由器模型 | `models/router/bge-m3/dynamic`          |


## 与知识库的对应关系

| **代码模块**               | **知识库环节**                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `load_data_in_batches`     | 数据预处理与chunk加载                |
| `Retriever`                | Retriever + Reranker流程               |
| `SequenceClassificationRouter`（domain） | Domain Router模块                    |
| `SequenceClassificationRouter`（dynamic） | Dynamic Router模块                  |
| `RAGModel`                 | 端到端Pipeline（DomainRouter→DynamicRouter→Reader） |
| `generate_predictions`     | 实验推理与结果记录                     |

```python
def initialize_batch():
    return {"interaction_id": [], "query": [], "search_results": [], "query_time": [], "answer": []}
```
**数据清洗相关**
**必需的JSONL格式**：
```json
{
    "interaction_id": "unique_sample_id_001",
    "query": "用户问题文本",
    "search_results": [检索结果列表],
    "query_time": "2024-01-01T12:00:00Z",
    "answer": "标准答案"
}
```

### **2. 文件格式要求**

- **支持格式**：`.jsonl`（每行一个JSON对象）或`.jsonl.bz2`（压缩格式）
- **编码**：UTF-8
- **每行结构**：独立的JSON对象，不是JSON数组


**清洗后的数据应该**：
- **Chunk化处理**：将长文档切分为512 token的片段
- **HTML去噪**：清理网页标签，提取纯文本内容  
- **元信息保留**：保存`doc_id`、`offset`等来源信息
- **格式标准化**：统一为JSONL格式，每个chunk作为`search_results`的元素

### **清洗流程建议**

**原始网页数据 → 清洗后的JSONL格式**：

```python
# 伪代码示例：数据清洗流程
def preprocess_raw_data(raw_html_docs, questions_answers):
    cleaned_data = []
    for qa_pair in questions_answers:
        # 1. 清洗HTML，提取文本chunk
        relevant_chunks = extract_and_chunk_html(
            raw_html_docs, 
            chunk_size=512, 
            query=qa_pair['question']
        )
        
        # 2. 构造标准格式
        cleaned_item = {
            "interaction_id": qa_pair['id'],
            "query": qa_pair['question'],
            "search_results": relevant_chunks,  # 预处理的候选chunk列表
            "query_time": qa_pair['timestamp'],
            "answer": qa_pair['ground_truth_answer']
        }
        cleaned_data.append(cleaned_item)
    
    return cleaned_data
```
