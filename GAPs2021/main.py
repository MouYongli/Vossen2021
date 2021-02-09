# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    from gapsnet.models import VGG16, FCN32s
    vgg16 = VGG16(pretrained=True)

    model = FCN32s()
    model.copy_params_from_vgg16(vgg16)
    for key in model.state_dict().keys():
        print(key)

