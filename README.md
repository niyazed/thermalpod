# Thermalpod

### ⚠️ DO NOT USE THIS COMMERCIALLY AS A PRODUCT.

I made this out of curiosity and experiment how this thing will look like. This is fully based on some image processing techniques using OpenCV, no hardware components are used. It can be integrated with <b>IR Camera</b> to get the actual temperature. Now, the temperatue is based on only some mathematical operation. So, make sure to integrate an IR and test before doing anything further.

Here I attached a youtube link - https://www.youtube.com/watch?v=k6qim96wB4k

#### Dependencies
- OpenCV
- NumPy
- Python3

#### Get Started
Run this command to start:
```sh
    $ python realtime_demo.py
```
There are two functions:
- Only Face
- Whole Frame

Comment `only_face()` if you want to work with whole frame.
```python
    if __name__ == "__main__":
        # whole_frame()
        only_face()
```

##### Note - If you want you can choose any other Face Detector, Here I am using `Haar Cascade` for simplicity.

#### Output
![output.gif](output.gif)


### Future Plan
IR sensor integration with a raspberry pi to give the project a whole look.

## License
MIT License

Copyright (c) 2020 Niyaz Bin Hashem

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

