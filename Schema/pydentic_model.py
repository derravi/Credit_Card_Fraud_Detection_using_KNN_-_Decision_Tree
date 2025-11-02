from pydantic import BaseModel,Field,computed_field
from typing import Annotated ,Literal

class UserInput(BaseModel):

    merchant:Annotated[str,Field(...,description="Enter the merchant.",examples=["fraud_Kirlin and Sons"])]
    category:Annotated[str,Field(...,description="Enter the category(e.g., travel, health_fitness).",examples=["travel"])]
    amt:Annotated[float,Field(...,gt=0,description="Enter the Transection Ammount",examples=["2000.00"])]
    gender:Annotated[Literal['M','F'],Field(...,description="Enter the Gender From 'M','F'.",examples=['M'])]
    city_pop:Annotated[int,Field(...,gt=0,description="Enter the City Population.",examples=[2000])]
    lat:Annotated[int,Field(...,gt=0,description="Enter the Latitude.",examples=[2041])]
    long:Annotated[float,Field(...,description="Enter the Longitude.",examples=[-125.35])]

    @computed_field
    @property
    def merch_lat(self) -> float:
        return self.lat + 0.01

    @computed_field
    @property
    def merch_long(self) -> float:
        return self.long + 0.01
    
