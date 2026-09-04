def get_c(c):
    match c.lower():
        case "san jose":
            return 37.3382, -121.8863
        case "san francisco":
            return 37.7749, -122.4194
        case "new york":
            return 40.7128, -74.0060
        case "los angeles":
            return 34.0522, -118.2437
        case _:
            return None


c = input("Enter a city: ")

a = get_c(c)

if a:
    x, y = a
    print(f"{c}: latitude={x}, longitude={y}")
else:
    print(f"Sorry, I don't know the coordinates for {c}.")
