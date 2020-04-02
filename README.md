# ndarray-inverse [![Build Status](https://travis-ci.org/tjpotts/ndarray-inverse.svg?branch=master)](https://travis-ci.org/tjpotts/ndarray-inverse)

Find the inverse `B` of a square matrix `A` such that `A*B=I`

## Example

```javascript
import ndarray from 'ndarray';
import zeros from 'zeros';
import inverse from 'ndarray-inverse';

const A = ndarray([1, 2, 3, 4], [2,2]);
const B = zeros([2,2]);
inverse(B,A);

console.log(show(B));
// -2.000    1.000
//  1.500   -0.500
```

## Installation

```javascript
$ npm install ndarray-inverse
```

## API

### success = ndarray-inverse(output, input)
**Arguments**:
- `output`: Output destination. The shape must mastch the shape of the input, otherwise an error will be thrown.
- `input`: Matrix to invert in the form of a 2-dimensional ndarray. An ndarray of dimension not equal to 2 or whose shape is not square will throw an error.

**Returns**: 'true' if the matrix inversion succeeded. 'false' if the matrix was not invertible.

## License
&copy; 2020 Timothy Potts. MIT License.

