**AI-Based Distributed Web Crawler with Semantic Ranking**

**Rishab Kalluri, Aarav Sharma, Maanas Baraya, Anish Bandari**

**Motivation and Objectives**

Web crawlers are a key part of the modern internet. Things such as search engines, data aggregators, and even knowledge bases all depend on web crawlers in order to download content from the web. Traditional crawlers work by following hyperlinks. They do this through navigating in a breadth first or best first manner. They also treat every found URL as roughly equal until some sort of link based signal (like PageRank) can be computed. This practice is largely wasteful. It’s inevitable for crawlers with limited bandwidth or time to spend large amounts of resources fetching irrelevant pages. This problem is made worse in things like focused crawling, in which the goal is to build an index of high quality for a specific domain. In short, the problem comes down to the fact that a traditional crawler can’t determine whether a URL leads to relevant content until after it has already fetched the page.

Through this project, we propose creating an AI based distributed web crawler. This crawler will use semantic understanding as part of its crawl prioritization. Each node within the crawler will use lightweight sentence embeddings to evaluate the relevance of a URL as compared to traditional methods like relying just on link structure. The key part of this is that the crawler will do this before it will fetch that specific URL, and it will score it based on things like anchor text, context, and referring page content.

Our main project objectives are:

1) Design a distributed crawling system. In this system, multiple nodes will coordinate to crawl in parallel. They will partition the URL space to avoid redundant fetches.  
2) Integrate a semantic prioritization module using sentence embeddings (use something like all-MiniLM-L6-v2). This will be done to rank URLs by their predicted relevance.  
3) Implement a URL distribution strategy that is aware of topics. Within this, URLs are routed by both hash partitioning and semantic likeliness to nodes’ topic clusters.  
4) Evaluate our semantic crawler against established baselines (breadth first, link based, etc.). Compare metrics like crawl efficiency, relevance, bandwidth, and more.

**Related Work**

Early crawlers like Google (Brin and Page, 1998\) or Mercator (Heydon and Najork, 1999\) used centralized architectures. These had a single coordinator. Cho and Garcia-Molina (2002) then proposed the idea of parallel crawlers. These used hash based URL partitioning. UbiCrawler (Boldi et al., 2014\) built upon this by creating a fully distributed crawler with consistent hashing.

Apoidea (Singh et al., 2003\) used protocols based on DHT, but it only achieved 18 URLs/sec. PeerCrawl (Padliya, Georgia Tech) used Gnutella and improved to 40 URLs/sec. This used 4 nodes but crashed after only 30 minutes due to memory issues. Loo et al. (2004) also experimented with crawling based on DHT. They had limited results as well, and none of these methods used context aware prioritization.

Chakrabarti et al. (1999) introduced focused crawling. This method used classifiers to predict topic relevance and Diligenti et al. (2000) extended this. They implemented crawling based on context and specifically used anchor text to predict relevance before fetching URLs. Both of these were centralized and both used keyword based classifiers instead of things like semantic embeddings.

The sentence transformers library (Reimers and Gurevych, 2019\) allows for computing dense vector representations. These capture semantic meanings in milliseconds. Furthermore, dense retrieval systems (Karpukhin et al., 2020\) have shown that designing crawlers based on embedding similarity is usually better than just keyword matching.

Overall, no existing system combines both distributed crawling and semantic prioritization based on embeddings. Our project fills this gap. We propose integrating modern sentence embeddings with a distributed crawler architecture.

**Proposed Work**

We are planning to design our system with four major components:  a Coordinator Service, multiple Crawler Nodes, a Semantic Prioritization Module, and a shared URL Frontier with deduplication. The semantic module will be designed as a pluggable component, meaning it can be enabled or disabled depending on the use case which allows us to make controlled comparisons against the base cases.

And our final architecture will follow a three tier structure:

1. **Tier 1:** This would be the Coordinator Service, where a lightweight service model would be implemented for the node registration, topic cluster assignment, and health monitoring. The service will not directly be involved in crawling.  
2. **Tier 2:** This would be the Crawler Nodes, currently we are thinking of implementing 4-8 nodes, these do operate independent of each other and each with three internal sub components. And these are a Semantic Prioritizer which scores incoming URLs by their relevance, a Local URL Frontier which sorts the URLs to be crawled, and a Downloader and Extractor pipeline that fetches pages and extracts outbound links. And all these nodes would communicate laterally to forward URLS that hash to another node’s partition.  
3. **Tier 3:** This would be a Shared State Layer, which contains a distributed Bloom filter for global URL deduplication, a Content Store for downloaded pages and their metadata, and a Metrics Collector for evaluation purposes.

So with these three tiers the data flow would follow this path, where tier 1 assigns each node a topic cluster and a hash range. And the downloader fetches the pages and the extractor pulls all the outbound URLS along with the context. Then the semantic prioritizer assigns a value to each URL, which gives them priority and high priority URLs will enter the local frontier. And then the URLs which belong to another node’s partition will be forwarded laterally. And finally all URLs will be checked against the global loom filter before entering any frontier.

The main part of this project is the semantic prioritization module. Where each node runs a local instance of all-MiniLM-L6-v2, which is 80 MB and under 10ms inference on CPU. When a URL is discovered, it's sent to the module and it computes an embedding of the URL’s context and then it compares it to the node’s topic using a cosine similarity. And then the score degenerated by this would help us assign a priority in the frontier. These topic centroids are initialized from seed descriptions and updated incrementally as the nodes crawl more relevant pages, which allow for the topic representation to evolve during the crawl. 

We plan to use the following modes for comparison. This helps us identify the strengths and weaknesses of each prioritization strategy.

1. **BFS baseline:** URLs are crawled in a discovery order, and this has no semantic scoring.  
2. **Link Priority baseline:** URLS are prioritized by their inbound link count discovered during the crawl, approximating PageRank.  
3. **Semantic Priority mode:** URLS are  prioritized by embedding-based relevance score.

We plan to use Python to implement this, as Python has a large collection of useful and relevant libraries. We plan to use asyncio/aiohttp for concurrent fetching, sentence-transformers for embeddings, Redis for inter-node communication and shared state, and Flask for the coordinator. And the target crawl domain is cs relevant, and seeded from arXiv CS pages, tech blogs, and university CS department sites.

**Requirements**

In order to effectively complete the methodology described, we must have some software and hardware setup. 

* Software Requirements:  We must have Python 3.10+ installed. Within python, we will use libraries for semantic embeddings (Sentence Transformers), concurrency (Redis), parsing HTML files (Beautiful Soup), and a frontend (Flask).  
* Hardware Requirements: We will use our 4 laptops to enable the web crawling experience to run quicker. The computation power will be run on our CPUs. 

**Schedule**

To successfully complete this project on time, we will make efforts to have concrete goals accomplished each week. The table below shows our timeline.

| Week | Accomplished |
| :---: | ----- |
| 1 | Complete a literature review. Finish environmental setup, including all dependencies. First draft of a system design. |
| 2 | Reviewed and completed system design. Implemented a single-node crawler. This crawler will be able to download a website, extract the text, and search the frontier of webpages.  |
| 3 | Implemented a distributed coordinator. This will include having multiple crawlers working on the same assignment, with a hash-based system to determine what crawlers search what webpages.  |
| 4 | Semantic embedding focused crawler. Have a score for the frontier webpages, and order the search of the frontier as a priority queue.  |
| 5 | Work on metric collections on the different stages of the systems we implemented.  |
| 6 | Run further experimentations on these various systems, including changing the number of nodes utilized. Create a draft of the final report. |
| 7 | Finalize the report, work on final presentation and demoing crawler.  |

**Evaluation and Testing Method**

Every crawl mode will be run under identical conditions: 

* Same seed URLs  
* Same number of nodes  
* Same crawl budget : 10,000 \- 50,000 pages  
* Same rate limiting

Each experiment will be repeated three times to account for variance in the results.

We will measure the following: 

**Harvest Rate:** For an allocation of N pages, what percent is relevant to the target topic? The relevance is determined by a classifier trained on manually labeled pages.

**Relevance at Depth**: The cumulative relevance against pages crawled. A good semantic crawler should front load all the relevant pages, and should have a dropoff

**Redundancy Rate:** The percentage of near duplicate pages found, measured with a MinHash similarity score.

**Bandwidth Efficiency:** The cumulative bytes download per relevant page discovered.

**Throughput and Scalability:** The pages per second, per node, as the count increases from 1 to 8\. This is a proxy for semantic scoring overhead remaining acceptable.

**Cross-Node Communication**: The total bytes exchanged between nodes to track coordination overhead.

**Bibliography**

1. S. Brin and L. Page. "The anatomy of a large-scale hypertextual Web search engine." Computer Networks and ISDN Systems, 30(1-7):107-117, 1998\.

2. A. Heydon and M. Najork. "Mercator: A Scalable, Extensible Web Crawler." Compaq Systems Research Center, 1999\.

3. J. Cho and H. Garcia-Molina. "Parallel crawlers." In Proc. of the 11th International World Wide Web Conference, 2002\.

4. P. Boldi, B. Codenotti, M. Santini, and S. Vigna. "UbiCrawler: A scalable fully distributed web crawler." Software: Practice and Experience, 34(8):711-726, 2004\.

5. A. Singh, M. Srivatsa, L. Liu, and T. Miller. "Apoidea: A Decentralized Peer-to-Peer Architecture for Crawling the World Wide Web." In Proc. of the SIGIR Workshop on Distributed Information Retrieval, August 2003\.

6. V. Padliya. "PeerCrawl." Masters Project, Georgia Institute of Technology.

7. B.T. Loo, S. Krishnamurthy, and O. Cooper. "Distributed Web Crawling over DHTs." UC Berkeley Technical Report UCB/CSD-4-1305, February 2004\.

8. S. Chakrabarti, M. van den Berg, and B. Dom. "Focused crawling: a new approach to topic-specific Web resource discovery." Computer Networks, 31(11-16):1623-1640, 1999\.

9. M. Diligenti, F. Coetzee, S. Lawrence, C.L. Giles, and M. Gori. "Focused Crawling Using Context Graphs." In Proc. of the 26th VLDB Conference, pp. 527-534, 2000\.

10. N. Reimers and I. Gurevych. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." In Proc. of EMNLP-IJCNLP, pp. 3982-3992, 2019\.

11. V. Karpukhin et al. "Dense Passage Retrieval for Open-Domain Question Answering." In Proc. of EMNLP, pp. 6769-6781, 2020\.

12. J. Cho, H. Garcia-Molina, and L. Page. "Efficient Crawling through URL Ordering." In Proc. of the 7th WWW Conference, pp. 161-172, 1998\.