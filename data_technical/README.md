# Testing the technical processing of the tool.

This directory contains sample data files that describe a graph. The File, promptFiles/technicalPrompt.txt is used with these data files to test the local model.

## Commands to use these files

- Rename this directory to `data/` (this may mean that a previous directory called `data/` will have to also be renamed as well.)
  - `mv data_technical data`
- `poetry run infomaid --resetdb --usetxt`
- `poetry run infomaid --useowndata --promptfile "promptFiles/technicalPrompt.txt"`

## Output

```
 STORYSEED: what is a bridge between points?

 RESPONSE:  Based on the given context, there are no direct bridges connecting any of the points. However, the sequence forms a cycle, and if we imagine a conceptual 'bridge' (or path) connecting each point in order (A-C-D-A), then this could be considered a circular route where each point is connected to the next one. But there are no explicitly stated bridges between points.

