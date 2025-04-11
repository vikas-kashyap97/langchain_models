from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class Student(BaseModel):

   # name: str
     name: str = 'Vikas'  # default value
     age: Optional[int] = None      # optional fields
     email: EmailStr
     cgpa: float =  Field(gt=0, lt=10, default=0, description='A decimal is representing the cgpa value of the student.')


#new_student = {'name':"Vikas"}
new_student = {"age" : 25, 'email':'abc@gmail.com'} # , new_student = {} # None


student = Student(**new_student)

print(student)