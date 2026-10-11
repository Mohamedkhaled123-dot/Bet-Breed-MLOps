from bet_breed_mlops.bento_model import register_bento_model


if __name__ == "__main__":
    model = register_bento_model()
    print(f"Registered BentoML model: {model.tag}")
