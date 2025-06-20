from PIL import Image
from typing import Any, Dict, List, Union, cast
from colpali_engine.models.InternVL2 import ColInternProcessor

class CustomCollator_InternVL2:
    def __init__(
        self,
        model_name,
        num_image_token: int = 256,
        max_length: int = 2048,
    ):
        self.processor = ColInternProcessor(model_name, num_image_token=num_image_token)
        self.num_image_token = num_image_token
        self.max_length = max_length

    def __call__(self, examples):
        
        texts_query: Union[List[str], List[None], List[Union[str, None]]] = []  # some documents don't have a query
        batch_doc = []
        batch_neg_doc = []
        batch_query = []

        # Process each example
        for example in examples:
            # texts_query.append(example["query"])

            batch_query.append(self.processor.process_queries(queries=[example["query"]]))

            if example["image"] is None:
                raise ValueError("Image is None - This collator does not support None images yet.")

            # Process the documents
            img = cast(Image, example["image"])
            batch_doc.append(self.processor.process_images(img=img))

            if "neg_image" in example and example["neg_image"] is not None:
                neg_img = cast(Image, example["neg_image"])
                batch_neg_doc.append(self.processor.process_images(img=neg_img))


        # # Process the queries
        # batch_query = None
        # if all([t is None for t in texts_query]):
        #     # print("All queries are `None`. Returning `None` for all queries.")
        #     pass
        # elif any([t is None for t in texts_query]):
        #     # If it's the first query that is not None but the rest are None, then it's hard negatives.
        #     raise ValueError("Some queries are None. This collator does not support None queries yet.")
        # else:
        #     texts_query = cast(List[str], texts_query)
        #     batch_query = self.processor.process_queries(
        #         queries=texts_query,
        #     )

        return [batch_query, batch_doc, batch_neg_doc]
        