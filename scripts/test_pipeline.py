from src.ingestion.pipeline import SimpleIngestionPipeline

pipeline = SimpleIngestionPipeline("machine learning")
chunks = pipeline.process()
