from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

text= """class Student:
    def __init__(self, name, roll_number, marks):
        self.name = name
        self.roll_number = roll_number
        self.marks = marks

    def display_info(self):
        print(f"Name: {self.name}")
        print(f"Roll Number: {self.roll_number}")
        print(f"Marks: {self.marks}")
            def calculate_grade(self):
        if self.marks >= 90:
            return "A"
        elif self.marks >= 75:
            return "B"
        else:
            return "F"

student1 = Student("Alice", 101, 92)
student2 = Student("Bob", 102, 67)
# Display info and grades
student1.display_info()
print(f"Grade: {student1.calculate_grade()}\n")

student2.display_info()
print(f"Grade: {student2.calculate_grade()}")

# Exlanation -This Python program defines a class called Student, which represents basic student information. It uses a constructor (__init__) to initialize the student’s name, roll number, and marks.
The class includes two main methods: display_info() to show the student’s details, and calculate_grade() to determine the grade based on marks using simple conditionals (A, B, F).We then create instances (objects) of the Student class with example data, and call the methods to display each student's information and grade. This is how object-oriented programming helps manage data clearly and modularly.
"""
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=500,
    chunk_overlap=0,
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks[0])