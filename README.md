# WiSARD implementation in Matlab

This code implements the Weightless Neural Network WiSARD (Wilkie,Stonham & Aleksander’s Recognition Device).

It allows the use of two different data structure for the discriminators: matrix or map.

For matrix, read and write accesses are instantaneous but it relies on the use of sparse matrixes (fast but uses a lot of memory).
For map, read and write take a bit more time, but it only saves positions used (slow but uses few memory).


The algorithm selects automatically between the two if no specification is given, based on the available ram criteria.


## Usage

```matlab
train_X = rand(1000, 100) > 0.5;
train_Y = rand(1000, 1) > 0.5;

test_X = rand(1000, 100) > 0.5;
test_Y = rand(1000, 1) > 0.5;

classes = {0 1};
nbits = 2;
input_size = 100;

W = WiSARD(classes, input_size, nbits);
W.fit(train_X, train_Y);
[labels, scores] = W.predict(test_X);

fprintf('Error: %.2f\n', 1 - sum(cell2mat(labels) == test_Y)/ length(test_Y) );
```

## Credits

This implementations was developed by Marco Simoes (msimoes@dei.uc.pt) and Bernardo Souza @ UFRJ.
All rights reserved.