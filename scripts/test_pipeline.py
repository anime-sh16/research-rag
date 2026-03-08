from src.ingestion.pipeline import SimpleIngestionPipeline

pipeline = SimpleIngestionPipeline(topics=["machine learning"])
summary = pipeline.process()
print(summary.model_dump_json(indent=2))
