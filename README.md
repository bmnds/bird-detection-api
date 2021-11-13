# bird-detection-api

* Dataset 315 Bird Species: https://www.kaggle.com/gpiosenka/

# instruções

1. Faça o download do modelo `resnet50.model.bruno.h5` a partir do notebook no Kaggle
    * [Link para o modelo no Kaggle](https://www.kaggle.com/bmendes00/bird-image-classification-using-cnns/data?select=resnet50.model.bruno.h5)

2. Para executar o projeto localmente, execute o script `main.py`:
    ```bash
    cd src
    python main.py
    ```

3. A aplicação ficará disponível em `http://localhost:5000`

4. Para testar a aplicação, faça um `POST` para o endpoint `/predict`:
    * defina o cabeçalho `X-API-KEY` com o valor `123`
    * envie o blob da imagem no parâmetro `image`
    
5. O resultado será um objeto `JSON` com a seguinte estrutura:
    ```JSON
    {
        "bird": "CANARY",
        "probability": 0.8888,
        "predictions": [
            "specie1": 0.0000,
            "specie2": 0.0000,
            ...
            "specieN": 0.0000,
        ]
    }
    
    ```

