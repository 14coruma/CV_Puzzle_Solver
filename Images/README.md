### Helpful info about how dataset was made
#### Imagemagick:
* Image resize:
```
convert <file> -resize 256x256^ -gravity <north,south,east,west,center> \
-extent 256x256 <file>
```

* Image rotate:
```
convert <file> -distort SRT <n degrees> <file>
```
