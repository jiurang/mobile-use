from pydantic import BaseModel, Field, model_validator

from minitap.mobile_use.utils.ui_hierarchy import ElementBounds


class Target(BaseModel):
    """
    A comprehensive locator for a UI element, supporting a fallback mechanism.
    """

    resource_id: str | None = Field(None, description="The resource-id of the element.")
    resource_id_index: int | None = Field(
        None,
        description="The zero-based index if multiple elements share the same resource-id.",
    )
    text: str | None = Field(
        None, description="The text content of the element (e.g., a label or placeholder)."
    )
    text_index: int | None = Field(
        None, description="The zero-based index if multiple elements share the same text."
    )
    bounds: ElementBounds | None = Field(
        None, description="The x, y, width, and height of the element."
    )
    ocr_index: int | None = Field(
        None,
        description=(
            "The 1-based [i] of a line in the 'Text on screen (OCR)' block, used verbatim. "
            "Resolves to that line's bounds, so it works for text the accessibility tree "
            "cannot see (video captions, canvas/surface content, custom-drawn UI). "
            "Do NOT confuse it with text_index/resource_id_index: those are 0-based and "
            "index the accessibility tree, not the OCR block."
        ),
    )

    @model_validator(mode="after")
    def _default_indices(self):
        # Treat empty strings like “not provided”
        if (
            self.resource_id is not None and self.resource_id != ""
        ) and self.resource_id_index is None:
            self.resource_id_index = 0
        if (self.text is not None and self.text != "") and self.text_index is None:
            self.text_index = 0
        return self
