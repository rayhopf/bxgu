

# 相似地址发现
## 问题定义
假设现在有这么一个需求：给定一组地址，我们希望找到，跟这些地址相关的其他地址。

比如，给定一组币安交易所的地址，我们希望找到跟币安相关的其他地址。

这个问题可以抽象成一个机器学习分类问题：给未知地址分类，跟币安相关的标记成 1，跟币安无关的标记成 0.


## 实现步骤
在数据和算力有限的情况下，我们先用特征工程和相似度来实现一个简单的版本。

这是下面的 SQL 涉及的步骤 (对应 SQL step 1 - step 8)：

1. 数据预处理（过滤时间和小额交易）, step 1
2. 特征提取（交易次数，交易额，交易对手数量）, step 2 - step 3
3. 归一化（min-max归一化）, step 4
4. 和给定一组地址计算相似度（余弦距离）, step 5 - step 6
5. 相似度平均值排序, step 7 - step 8

期望的结果是，相似度高的就是与给定地址更相关的地址。

### DUNE SQL

https://dune.com/queries/4010794

```sql
WITH
-- Step 1: Filter relevant transfers for USDT ERC20 after 2024-08-01
relevant_transfers AS (
    SELECT 
        "from",
        "to",
        value / 1e6 AS value_decimal -- Adjusting for decimal places to treat as regular floats
    FROM 
        erc20_ethereum.evt_transfer
    WHERE 
        contract_address = 0xdac17f958d2ee523a2206206994597c13d831ec7
        AND evt_block_time > TRY_CAST(CAST('2024-08-01 00:00' AS TIMESTAMP) AS TIMESTAMP)
        AND value > 1e9
),

-- Step 2: Aggregate by (from, to) pairs
aggregated_transfers AS (
    SELECT
        "from",
        "to",
        COUNT(*) AS transfer_count,         -- Count of transactions between each (from, to) pair
        SUM(value_decimal) AS total_amount  -- Total amount transferred between each (from, to) pair
    FROM 
        relevant_transfers
    GROUP BY
        "from", "to"
),

-- Step 3: Calculate behavioral features for each address
address_features AS (
    SELECT 
        a.address,
        TRY_CAST(SUM(agg.transfer_count) AS DOUBLE) AS total_transfer_count, -- Total number of transfers for each address
        TRY_CAST(SUM(agg.total_amount) AS DOUBLE) AS total_transfer_amount, -- Total value of transfers for each address
        TRY_CAST(COUNT(DISTINCT CASE 
            WHEN agg."from" = a.address THEN agg."to" 
            WHEN agg."to" = a.address THEN agg."from" 
        END) AS DOUBLE) AS degree -- Number of unique counterparts
    FROM (
        SELECT 
            "from" AS address 
        FROM aggregated_transfers
        UNION 
        SELECT 
            "to" AS address 
        FROM aggregated_transfers
    ) a
    JOIN aggregated_transfers agg 
    ON agg."from" = a.address OR agg."to" = a.address
    GROUP BY a.address
),

-- Step 4: Normalize features using min-max scaling
feature_min_max AS (
    SELECT
        MIN(degree) AS min_degree,
        MAX(degree) AS max_degree,
        MIN(total_transfer_count) AS min_transfer_count,
        MAX(total_transfer_count) AS max_transfer_count,
        MIN(total_transfer_amount) AS min_transfer_amount,
        MAX(total_transfer_amount) AS max_transfer_amount
    FROM 
        address_features
),

normalized_address_features AS (
    SELECT 
        af.address,
        (af.degree - fmm.min_degree) / NULLIF(fmm.max_degree - fmm.min_degree, 0) AS norm_degree,
        (af.total_transfer_count - fmm.min_transfer_count) / NULLIF(fmm.max_transfer_count - fmm.min_transfer_count, 0) AS norm_transfer_count,
        (af.total_transfer_amount - fmm.min_transfer_amount) / NULLIF(fmm.max_transfer_amount - fmm.min_transfer_amount, 0) AS norm_transfer_amount
    FROM 
        address_features af
    CROSS JOIN 
        feature_min_max fmm
),

-- Step 5: Define target addresses (replace 'a1', 'a2', 'a3' with actual addresses)
target_addresses AS (
    SELECT 
        address, 
        norm_degree, 
        norm_transfer_count, 
        norm_transfer_amount
    FROM 
        normalized_address_features
    WHERE 
        address IN (
            0x345d8e3a1f62ee6b1d483890976fd66168e390f2,
            0xf60c2ea62edbfe808163751dd0d8693dcb30019c,
            0x56eddb7aa87536c09ccc2793473599fd21a8b17f,
            0x892e9e24aea3f27f4c6e9360e312cce93cc98ebe,
            0x34ea4138580435b5a521e460035edb19df1938c1,
            0x61189da79177950a7272c88c6058b96d4bcd6be2
        ) -- Replace these with actual addresses
),

-- Step 6: Compute similarity scores for each address with each target address
similarity_with_target AS (
    SELECT 
        naf.address AS address,
        ta.address AS target_address,
        (   (naf.norm_degree * ta.norm_degree) + 
            (naf.norm_transfer_count * ta.norm_transfer_count) + 
            (naf.norm_transfer_amount * ta.norm_transfer_amount)
        ) / (
            SQRT(POWER(naf.norm_degree, 2) + 
                 POWER(naf.norm_transfer_count, 2) + 
                 POWER(naf.norm_transfer_amount, 2))
            *
            SQRT(POWER(ta.norm_degree, 2) + 
                 POWER(ta.norm_transfer_count, 2) + 
                 POWER(ta.norm_transfer_amount, 2))
        ) AS similarity_score
    FROM 
        normalized_address_features naf
    CROSS JOIN 
        target_addresses ta
),

-- Step 7: Average similarity scores across all target addresses
average_similarity AS (
    SELECT 
        address,
        AVG(similarity_score) AS average_similarity_score
    FROM 
        similarity_with_target
    GROUP BY 
        address
)

-- Step 8: Final sorted results
SELECT 
    address,
    average_similarity_score AS similarity_score
FROM 
    average_similarity
ORDER BY 
    similarity_score DESC
```


### 结果片段

运行上面的 SQL 耗时 14 分钟，结果片段如下：

| address                                    | similarity_score       |
|--------------------------------------------|------------------------|
| 0xf63fba4ba747e0eb48c41e37688ed8f2176da70b | NaN                    |
| 0x8b882f888d3dae2516fef61f2b3bdcc00762a705 | NaN                    |
| 0x990496a4796d719db3d683e3c988b7164b81a87b | NaN                    |
| 0xfe1e46588af85b2489f1d3c401809b4fec5c966b | NaN                    |
| 0xb64d9784e8516983243434ce3badf967fd5cc71e | 0.9491032958989143     |
| 0x4500ef9f85dc91b5b9486894ad4abb26a5a954af | 0.9491006640305004     |
| 0xb068ec5fb656aee1c646c33b49bda288880d3c0f | 0.9490931482463221     |
| 0x9cfb73f0c8b494aaf8f00da25cd32286cf56ac98 | 0.9490808749048989     |
| 0x22429d16c6615900d4dbcada6bead0bb37930fb0 | 0.9490753628361498     |
| 0xd3e7927bdfff1f1b186399a0c07e8e48f30c3548 | 0.9490161958656362     |
| 0x7cd4563f4ffc1320d869e1f4c72e01a87598e2b2 | 0.9489737401225152     |

target_addresses 是我找的几个 binance 的地址。
similarity_score 就是其他地址和 target_addresses 每个地址求完相似度然后取平均值。
结果中有一些 NaN 不确定什么原因。

相似度排名靠前的几个地址有交易所的地址，但是大部分地址和币安不相关，说明这个方法有效果，但是效果不好。
- 0xb64d9784e8516983243434ce3badf967fd5cc71e: Tokenize Xchange 1
- 0x7cd4563f4ffc1320d869e1f4c72e01a87598e2b2: matrixport


## 后续改进方案
1. 这个例子里只有 3 个特征，可以增加更多的特征来提升效果。
2. 如果数据和算力充足，可以试试图神经网络的方法。