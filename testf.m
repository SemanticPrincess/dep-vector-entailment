function a = testf (b, varargin)
length(varargin)
a = varargin{1}.isKey('abc')
varargin
end