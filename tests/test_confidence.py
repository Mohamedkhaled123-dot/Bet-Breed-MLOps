import torch
import pytest

from bet_breed_mlops.confidence import (
    apply_temperature,
    predict_with_confidence,
    predict_image,
)


def test_temperature_scaling():

    logits = torch.tensor(
        [2.0, 1.0, 0.0]
    )

    scaled = apply_temperature(
        logits,
        temperature=0.5,
    )

    expected = torch.tensor(
        [4.0, 2.0, 0.0]
    )

    assert torch.allclose(
        scaled,
        expected,
    )


def test_temperature_must_be_positive():

    logits = torch.tensor(
        [2.0, 1.0, 0.0]
    )

    with pytest.raises(ValueError):

        apply_temperature(
            logits,
            temperature=0.0,
        )


def test_high_confidence_prediction():

    logits = torch.tensor(
        [5.0, 0.0, 0.0]
    )

    class_names = {
        0: "Persian",
        1: "Bulldog",
        2: "Beagle",
    }

    result = predict_with_confidence(
        logits=logits,
        temperature=1.0,
        threshold=0.80,
        class_names=class_names,
    )

    assert result.class_index == 0
    assert result.breed == "Persian"
    assert result.is_confident is True
    assert result.confidence >= 0.80


def test_low_confidence_returns_not_sure():

    logits = torch.tensor(
        [1.0, 1.0, 1.0]
    )

    class_names = {
        0: "Persian",
        1: "Bulldog",
        2: "Beagle",
    }

    result = predict_with_confidence(
        logits=logits,
        temperature=1.0,
        threshold=0.80,
        class_names=class_names,
    )

    assert result.is_confident is False
    assert result.breed is None


def test_threshold_must_be_valid():

    logits = torch.tensor(
        [5.0, 0.0, 0.0]
    )

    class_names = {
        0: "Persian",
        1: "Bulldog",
        2: "Beagle",
    }

    with pytest.raises(ValueError):

        predict_with_confidence(
            logits=logits,
            temperature=1.0,
            threshold=1.5,
            class_names=class_names,
        )


def test_predict_image():
    from PIL import Image
    from unittest.mock import patch

    class DummyModel(torch.nn.Module):

        def forward(self, x):
            return torch.tensor(
                [[5.0, 0.0, 0.0]],
                dtype=torch.float32,
            )

    image = Image.new(
        "RGB",
        (224, 224),
        color="white",
    )

    class_names = {
        0: "Persian",
        1: "Bulldog",
        2: "Beagle",
    }

    model = DummyModel()

    with patch(
        "bet_breed_mlops.confidence.load_temperature",
        return_value=1.0,
    ), patch(
        "bet_breed_mlops.confidence.load_class_names",
        return_value=class_names,
    ):

        result = predict_image(
            model=model,
            image=image,
            threshold=0.80,
        )

    assert result.class_index == 0
    assert result.breed == "Persian"
    assert result.is_confident is True
    assert result.confidence >= 0.80