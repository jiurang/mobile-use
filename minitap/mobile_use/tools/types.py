from pydantic import BaseModel, Field, model_validator

from minitap.mobile_use.utils.ui_hierarchy import ElementBounds


class Target(BaseModel):
    """
    A comprehensive locator for a UI element, supporting a fallback mechanism.
    """

    index: int | None = Field(
        None,
        description=(
            "The 1-based [i] of a row in the 'Screen elements' block, used verbatim. "
            "One numbering covers both the element tree and the OCR text, so an "
            "index alone identifies the element - the model never has to copy a "
            "resource-id or coordinates."
        ),
    )
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
            "Legacy: the 1-based [i] of a line in the 'Text on screen (OCR)' block. "
            "When a screen element table is present it is an alias of `index`; "
            "otherwise it indexes the OCR results directly. "
            "Do NOT confuse it with text_index/resource_id_index: those are 0-based and "
            "index the accessibility tree, not the OCR block."
        ),
    )
    vision: str | None = Field(
        None,
        description=(
            "A natural-language description of the target, for things neither the "
            "element table nor the OCR block can name (icons, images, canvas). The "
            "framework locates it on a screenshot and supplies the coordinates; never "
            "pair it with your own bounds. Prefer `index` whenever a row exists."
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
