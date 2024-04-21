-- MySQL dump 10.13  Distrib 8.3.0, for macos14.2 (arm64)
--
-- Host: localhost    Database: test
-- ------------------------------------------------------
-- Server version	8.3.0

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `concepts`
--

DROP TABLE IF EXISTS `concepts`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `concepts` (
  `concept_id` int NOT NULL AUTO_INCREMENT,
  `parent_id` int DEFAULT NULL,
  `file_id` int DEFAULT NULL,
  `concept_name` varchar(255) DEFAULT NULL,
  `concept_page` varchar(255) DEFAULT NULL,
  `concept_description` text,
  PRIMARY KEY (`concept_id`),
  KEY `parent_id` (`parent_id`),
  KEY `file_id` (`file_id`),
  CONSTRAINT `concepts_ibfk_1` FOREIGN KEY (`parent_id`) REFERENCES `concepts` (`concept_id`),
  CONSTRAINT `concepts_ibfk_2` FOREIGN KEY (`file_id`) REFERENCES `files` (`file_id`)
) ENGINE=InnoDB AUTO_INCREMENT=168 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `concepts`
--

LOCK TABLES `concepts` WRITE;
/*!40000 ALTER TABLE `concepts` DISABLE KEYS */;
INSERT INTO `concepts` VALUES (1,NULL,1,'Overview of NLP','3',NULL),(2,NULL,1,'Historical Background','9',NULL),(3,NULL,1,'Approaches to NLP','10',NULL),(4,NULL,1,'Preprocessing Techniques','11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29',NULL),(5,NULL,2,'Introduction to Information Extraction','3',NULL),(6,NULL,2,'Named Entity Recognition (NER)','5, 6, 7, 8',NULL),(7,NULL,2,'Part-Of-Speech Tagging','13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26',NULL),(8,NULL,2,'Dependency Parsing','31, 32, 33',NULL),(9,NULL,3,'Term Weighting Schemes','2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14',NULL),(10,NULL,3,'Topic Modeling','15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42',NULL),(11,NULL,3,'Dimensionality Reduction','42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57',NULL),(12,NULL,4,'Text Classification','2, 3, 4, 5, 6',NULL),(13,NULL,4,'Clustering','31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47',NULL),(14,NULL,5,'Evaluation Metrics','4, 5, 6, 7, 8, 9',NULL),(15,NULL,5,'Word Embeddings','33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54',NULL),(16,NULL,6,'Sequential Data','3, 4, 5',NULL),(17,NULL,6,'RNNs','6, 7, 8',NULL),(18,NULL,6,'LSTMs','15, 16, 17, 18, 19, 20, 21',NULL),(19,NULL,6,'GRUs','22, 23, 24',NULL),(20,NULL,6,'Bi-Directional RNNs','25',NULL),(21,NULL,7,'Seq2Seq Models','3, 4',NULL),(22,NULL,7,'Attention Mechanism','5, 6, 7, 9',NULL),(23,NULL,7,'Transformer Models','10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24',NULL),(24,NULL,8,'K-Fold Cross Validation','',NULL),(25,NULL,8,'Optimisers','10, 11, 12, 13, 14, 15',NULL),(26,NULL,8,'Loss Functions','17, 18, 19',NULL),(27,NULL,8,'Batch Size and Learning Rate','21, 22, 23, 24',NULL),(28,NULL,8,'Epochs and Early Stopping','25, 26',NULL),(29,NULL,8,'Gradient Clipping','27, 28, 29',NULL),(30,NULL,8,'Regularisation Techniques','30, 31, 32',NULL),(31,NULL,9,'Transformer Recap','',NULL),(32,NULL,9,'Pre Training Objectives','',NULL),(33,NULL,9,'Autoregressive Models','9',NULL),(34,NULL,9,'Autoencoding Models','10, 11',NULL),(35,NULL,9,'Positional Encodings','12',NULL),(36,NULL,9,'Tokenizers','',NULL),(37,NULL,9,'Pre-training Process','15',NULL),(38,NULL,9,'Fine-tuning Process','16, 17',NULL),(39,NULL,9,'BERT Variants','',NULL),(40,NULL,9,'GPT Series','',NULL),(41,NULL,9,'T5 and BART Models','26, 27',NULL),(42,NULL,9,'XLNet','28, 29',NULL),(43,NULL,10,'Overview','3',NULL),(44,NULL,10,'Healthcare','4, 5',NULL),(45,NULL,10,'Finance and Banking','6, 7',NULL),(46,NULL,10,'Retail and E-Commerce','8, 9',NULL),(47,NULL,10,'Legal Industry','10, 11',NULL),(48,NULL,10,'Automotive Industry','12, 13',NULL),(49,NULL,10,'Publishing Industry','14, 15',NULL),(50,NULL,10,'Education','16, 17',NULL),(51,NULL,10,'Travel and Hospitality','18, 19',NULL),(52,NULL,10,'Media and Entertainment','20, 21',NULL),(53,NULL,10,'Government & Public Sector','22, 23',NULL),(54,NULL,11,'Fraud Detection in E-commerce','',NULL),(55,NULL,11,'Human Trafficking Detection','8, 9, 10, 11, 12',NULL),(56,NULL,11,'ChatGPT-based Applications','',NULL),(57,1,1,'Definition of NLP','3',NULL),(58,1,1,'Applications of NLP','3',NULL),(59,2,1,'Early developments','9',NULL),(60,2,1,'RNNs and early neural models','9',NULL),(61,3,1,'Symbolic','10',NULL),(62,3,1,'Stochastic','10',NULL),(63,3,1,'Neural Models','10',NULL),(64,4,1,'RegEx','12, 13, 14, 15, 16, 17',NULL),(65,4,1,'Tokenization','19, 20',NULL),(66,4,1,'Stemming','21, 22, 23',NULL),(67,4,1,'Lemmatization','24, 25, 26, 27',NULL),(68,4,1,'Bag-Of-Words','30, 31, 32, 33, 34',NULL),(69,4,1,'N-grams','35, 36, 37',NULL),(70,5,2,'Definition and Importance','3',NULL),(71,5,2,'Applications','3',NULL),(72,6,2,'Fundamentals of NER','5, 6',NULL),(73,6,2,'Implementing NER with spaCy','8, 9',NULL),(74,7,2,'Overview of POS Tagging','13, 14',NULL),(75,7,2,'Hidden Markov Models (HMMs)','16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26',NULL),(76,8,2,'Introduction to Dependency Parsing','31',NULL),(77,8,2,'Worked Example in spaCy','32, 33',NULL),(78,9,3,'TF-IDF','3, 4, 5, 6, 7',NULL),(79,9,3,'BM25','11, 12, 13, 14',NULL),(80,10,3,'LDA','15, 16, 17, 18, 19, 20, 21, 22, 23, 24',NULL),(81,10,3,'LSA','25, 26, 27, 28, 29, 30',NULL),(82,10,3,'pLSA','31, 32',NULL),(83,10,3,'NMF','34, 35, 36, 37, 38, 39, 40, 41',NULL),(84,11,3,'PCA','43, 44, 45, 46, 47, 48, 49, 50, 51',NULL),(85,11,3,'SVD','52, 53, 54, 55, 56, 57',NULL),(86,12,4,'Methods','5',NULL),(87,12,4,'Applications','2, 3, 4, 48',NULL),(88,13,4,'K-Means','31, 32, 33, 34, 35, 36, 37',NULL),(89,13,4,'Hierarchical Clustering','38, 39, 40, 41, 42, 43, 44, 45',NULL),(90,13,4,'Fuzzy Clustering','46, 47',NULL),(91,14,5,'Confusion Matrix','6, 7',NULL),(92,14,5,'Precision and Recall','8',NULL),(93,14,5,'Micro and Macro Metrics','9, 10',NULL),(94,15,5,'Word2Vec','35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49',NULL),(95,15,5,'GloVe','50, 51, 52, 53, 54',NULL),(96,16,6,'Characteristics of Sequential Data','3',NULL),(97,16,6,'Challenges with Traditional Models','4',NULL),(98,16,6,'Modern Techniques for Sequential Data','5',NULL),(99,17,6,'Vanilla RNNs','6, 7',NULL),(100,17,6,'Activation Functions','9',NULL),(101,17,6,'RNN Classifiers','11',NULL),(102,18,6,'Cell States','17',NULL),(103,18,6,'Forget Gates','18',NULL),(104,18,6,'Input Gates','19',NULL),(105,18,6,'Output Gates','21',NULL),(106,19,6,'Reset Gates','23',NULL),(107,19,6,'Update Gates','24',NULL),(108,21,7,'Encoder-Decoder Architecture','3, 4',NULL),(109,21,7,'Attention Mechanism Necessity','5',NULL),(110,22,7,'General Attention Mechanism','6',NULL),(111,22,7,'Attention Scores Calculation','9',NULL),(112,23,7,'Core Components','11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24',NULL),(113,24,8,'Importance of proper k value','6',NULL),(114,24,8,'Stratified K-Fold example','7',NULL),(115,25,8,'Gradient Descent','11',NULL),(116,25,8,'Stochastic Gradient Descent (SGD)','12',NULL),(117,25,8,'ADAM','14',NULL),(118,26,8,'Binary Cross Entropy','18',NULL),(119,26,8,'Categorical Cross Entropy','19',NULL),(120,31,9,'Encoder-Decoder Architecture','3',NULL),(121,31,9,'Transformer-XL','4',NULL),(122,32,9,'Pre-training and Fine-tuning','5, 6',NULL),(123,32,9,'Unified Downstream Tasks','7',NULL),(124,32,9,'Objective Types: AR and AE','8',NULL),(125,36,9,'Sub-word Tokenization','13, 14',NULL),(126,39,9,'RoBERTa','18',NULL),(127,39,9,'DistilBERT','19',NULL),(128,39,9,'DistilRoBERTa','20',NULL),(129,40,9,'GPT-1 to GPT-3','21, 22',NULL),(130,40,9,'GPT-4','23',NULL),(131,40,9,'In-context Learning and Fine-tuning','24, 25',NULL),(132,43,10,'Industries','3',NULL),(133,44,10,'Clinical Documentation','4',NULL),(134,44,10,'Diagnostic Assistance','4',NULL),(135,44,10,'Drug Discovery and Development','4',NULL),(136,45,10,'Financial Report Automation','6',NULL),(137,45,10,'Fraud Detection','6',NULL),(138,45,10,'Customer Service Chatbots','6',NULL),(139,46,10,'Sentiment Analysis for Product Reviews','8',NULL),(140,46,10,'Personalized Recommendations','8',NULL),(141,46,10,'Customer Service Chatbots','8',NULL),(142,47,10,'Document Review and Analysis','10',NULL),(143,47,10,'Predictive Analysis','10',NULL),(144,47,10,'Compliance Monitoring','10',NULL),(145,48,10,'Voice-Activated Assistants','12',NULL),(146,48,10,'Customer Feedback Analysis','12',NULL),(147,48,10,'Automated Vehicle Information Support','12',NULL),(148,49,10,'Content Curation and Management','14',NULL),(149,49,10,'Reader Engagement Analysis','14',NULL),(150,49,10,'Automated Content Generation','14',NULL),(151,50,10,'Plagiarism Detection','16',NULL),(152,50,10,'Content Personalization','16',NULL),(153,50,10,'Language Learning Tools','16',NULL),(154,51,10,'Automated Booking Assistants','18',NULL),(155,51,10,'Customer Feedback Analysis','18',NULL),(156,51,10,'Language Translation Services','18',NULL),(157,52,10,'Content Recommendation Systems','20',NULL),(158,52,10,'Automated Subtitling and Dubbing','20',NULL),(159,52,10,'Sentiment Analysis of Social Media','20',NULL),(160,53,10,'Public Sentiment Analysis','22',NULL),(161,53,10,'Automated Public Services','22',NULL),(162,53,10,'Document and Record Management','22',NULL),(163,54,11,'Fulfillment Fraud Detection','4, 5, 6',NULL),(164,54,11,'Off-platform Fraud Detection','',NULL),(165,55,11,'Using Banking Data for Detection','',NULL),(166,56,11,'Content Moderation','13, 14',NULL),(167,56,11,'Product Recommendation Chatbot','20, 21, 22, 23, 24, 25',NULL);
/*!40000 ALTER TABLE `concepts` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `courses`
--

DROP TABLE IF EXISTS `courses`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `courses` (
  `course_id` int NOT NULL AUTO_INCREMENT,
  `course_code` varchar(50) DEFAULT NULL,
  `course_name` varchar(255) DEFAULT NULL,
  `course_description` text,
  PRIMARY KEY (`course_id`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `courses`
--

LOCK TABLES `courses` WRITE;
/*!40000 ALTER TABLE `courses` DISABLE KEYS */;
INSERT INTO `courses` VALUES (1,'EE6405','Natural Language Processing',NULL);
/*!40000 ALTER TABLE `courses` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `examples`
--

DROP TABLE IF EXISTS `examples`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `examples` (
  `example_id` int NOT NULL AUTO_INCREMENT,
  `concept_id` int DEFAULT NULL,
  `example_name` varchar(255) DEFAULT NULL,
  `example_page` varchar(255) DEFAULT NULL,
  `example_description` text,
  PRIMARY KEY (`example_id`),
  KEY `concept_id` (`concept_id`),
  CONSTRAINT `examples_ibfk_1` FOREIGN KEY (`concept_id`) REFERENCES `concepts` (`concept_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `examples`
--

LOCK TABLES `examples` WRITE;
/*!40000 ALTER TABLE `examples` DISABLE KEYS */;
/*!40000 ALTER TABLE `examples` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `files`
--

DROP TABLE IF EXISTS `files`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `files` (
  `file_id` int NOT NULL AUTO_INCREMENT,
  `course_id` int DEFAULT NULL,
  `file_name` varchar(255) DEFAULT NULL,
  `title` varchar(255) DEFAULT NULL,
  `file_type` varchar(50) DEFAULT NULL,
  `file_path` varchar(255) DEFAULT NULL,
  `teaching_week` int DEFAULT NULL,
  `creation_date` datetime DEFAULT NULL,
  PRIMARY KEY (`file_id`),
  KEY `course_id` (`course_id`),
  CONSTRAINT `files_ibfk_1` FOREIGN KEY (`course_id`) REFERENCES `courses` (`course_id`)
) ENGINE=InnoDB AUTO_INCREMENT=12 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `files`
--

LOCK TABLES `files` WRITE;
/*!40000 ALTER TABLE `files` DISABLE KEYS */;
INSERT INTO `files` VALUES (1,1,'EE6405_W1_Introduction_to_NLP','Introduction to Natural Language Processing','slides','slides/EE6405_W1_Introduction_to_NLP',1,'2024-04-17 16:16:29'),(2,1,'EE6405_W2_Linguistic Analysis and Information Extraction','Linguistic Analysis and Information Extraction','slides','slides/EE6405_W2_Linguistic Analysis and Information Extraction',2,'2024-04-17 16:16:29'),(3,1,'EE6405_W3_Term Weighting Scheme and Topic Modelling_For Students','Term Weighting Scheme and Topic Modelling','slides','slides/EE6405_W3_Term Weighting Scheme and Topic Modelling_For Students',3,'2024-04-17 16:16:29'),(4,1,'EE6405_W4_Traditional ML and NLP Applications','Traditional Machine Learning Methods and NLP Applications','slides','slides/EE6405_W4_Traditional ML and NLP Applications',4,'2024-04-17 16:16:29'),(5,1,'EE6405_W5_EMaWE','Evaluation Metrics and Word Embeddings','slides','slides/EE6405_W5_EMaWE',5,'2024-04-17 16:16:29'),(6,1,'EE6405_W6_Neural Language Models','Neural Language Models','slides','slides/EE6405_W6_Neural Language Models',6,'2024-04-17 16:16:29'),(7,1,'EE6405_W7_Transformers','Transformers','slides','slides/EE6405_W7_Transformers',7,'2024-04-17 16:16:29'),(8,1,'EE6405_W8_HyperParameter Tuning','HyperParameter Tuning','slides','slides/EE6405_W8_HyperParameter Tuning',8,'2024-04-17 16:16:29'),(9,1,'EE6405_W9_Transformer Based LLMs','Transformer Based Large Language Models','slides','slides/EE6405_W9_Transformer Based LLMs',9,'2024-04-17 16:16:29'),(10,1,'EE6405_W10_NLP_Applications_Across_Industries','A Survey of NLP Applications Across Diverse Industries','slides','slides/EE6405_W10_NLP_Applications_Across_Industries',10,'2024-04-17 16:16:29'),(11,1,'EE6405_W11_Deep_dive_into_NLP','Deep-dive into NLP','slides','slides/EE6405_W11_Deep_dive_into_NLP',11,'2024-04-17 16:16:29');
/*!40000 ALTER TABLE `files` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `summaries`
--

DROP TABLE IF EXISTS `summaries`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `summaries` (
  `summary_id` int NOT NULL AUTO_INCREMENT,
  `file_id` int DEFAULT NULL,
  `key_points` text,
  `summary_page` varchar(50) DEFAULT NULL,
  PRIMARY KEY (`summary_id`),
  KEY `file_id` (`file_id`),
  CONSTRAINT `summaries_ibfk_1` FOREIGN KEY (`file_id`) REFERENCES `files` (`file_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `summaries`
--

LOCK TABLES `summaries` WRITE;
/*!40000 ALTER TABLE `summaries` DISABLE KEYS */;
/*!40000 ALTER TABLE `summaries` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2024-04-17 16:41:47
