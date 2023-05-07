import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# App creation and model loading
app = FastAPI()
model = joblib.load("./model.joblib")


class PersonStatus(BaseModel):
    """
    Input features validation for the ML model
    """
    HomePlanet: int
    CryoSleep: int
    Destination: int
    VIP: int
    spend_money: int
    Deck: int
    Side: int
    age_cat: int
    Cabin_region1: int
    Cabin_region2: int
    Cabin_region3: int
    Cabin_region4: int
    Cabin_region5: int
    Cabin_region6: int
    Cabin_region7: int


@app.post('/predict')
def predict(person: PersonStatus):
    """
    :param person: input data from the post request
    :return: predicted iris type
    """
    features = [[
        person.HomePlanet, person.CryoSleep, person.Destination, person.VIP,
        person.spend_money, person.Deck, person.Side, person.age_cat,
        person.Cabin_region1, person.Cabin_region2, person.Cabin_region3, person.Cabin_region4,
        person.Cabin_region5, person.Cabin_region6, person.Cabin_region7
                ]]
    prediction = model.predict(features).tolist()[0]
    return {
        "prediction": prediction
    }


if __name__ == '__main__':
    # Run server using given host and port
    uvicorn.run(app, host='localhost', port=80)
