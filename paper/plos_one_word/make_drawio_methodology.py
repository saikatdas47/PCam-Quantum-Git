"""Editable methodology figure and matching vector/raster exports from executed scope."""
from pathlib import Path
from xml.etree.ElementTree import Element,SubElement,ElementTree
from html import escape
import math
from PIL import Image,ImageDraw,ImageFont
P=Path('paper/PLOS_ONE_submission/figures');P.mkdir(parents=True,exist_ok=True)
W,H=1000,1140;scale=2.96
im=Image.new('RGB',(round(W*scale),round(H*scale)),'white');dr=ImageDraw.Draw(im)
root=Element('mxfile',host='draw.io',version='29.0.0');diagram=SubElement(root,'diagram',name='Executed study methodology',id='pcam-methodology');model=SubElement(diagram,'mxGraphModel',dx='1000',dy='1140',grid='0',gridSize='10',guides='1',tooltips='1',connect='1',arrows='1',fold='1',page='1',pageScale='1',pageWidth=str(W),pageHeight=str(H),math='0',shadow='0',background='#FFFFFF');rt=SubElement(model,'root');SubElement(rt,'mxCell',id='0');SubElement(rt,'mxCell',id='1',parent='0')
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="7.4in" height="8.436in" viewBox="0 0 {W} {H}"><rect width="1000" height="1140" fill="white"/>']
from matplotlib.font_manager import FontProperties,findfont
fontpath=findfont(FontProperties(family=['Arial','Liberation Sans','DejaVu Sans']))
boldpath=findfont(FontProperties(family=['Arial','Liberation Sans','DejaVu Sans'],weight='bold'));counter=0
nodes={}
def cell(value,style,x,y,w,h,id=None):
 global counter
 counter+=1;id=id or f'n{counter}'
 e=SubElement(rt,'mxCell',id=id,value=value,style=style,vertex='1',parent='1');SubElement(e,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),**{'as':'geometry'});return id

def text(lines,x,y,w,h,size=18,bold=False,align='center',color='#222222',id=None):
 if isinstance(lines,str):lines=lines.split('\n')
 value='\n'.join(lines)
 id=cell(value,f'text;html=0;whiteSpace=wrap;overflow=hidden;align={align};verticalAlign=middle;fontFamily=Arial;fontSize={size};fontColor={color};fontStyle={1 if bold else 0};spacing=0;strokeColor=none;fillColor=none;',x,y,w,h,id)
 f=ImageFont.truetype(boldpath if bold else fontpath,round(size*scale));lh=size*1.25;top=y+(h-lh*len(lines))/2
 for k,line in enumerate(lines):
  length=dr.textlength(line,font=f)/scale
  assert length<=w+2,(line,length,w)
  xx=x+(w-length)/2 if align=='center' else x
  yy=top+k*lh
  dr.text((round(xx*scale),round(yy*scale)),line,font=f,fill=color,anchor='lt')
  sx=x+w/2 if align=='center' else x;anchor='middle' if align=='center' else 'start'
  svg.append(f'<text x="{sx}" y="{yy+size*.86}" font-family="Arial" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}" text-anchor="{anchor}">{escape(line)}</text>')
 return id

def rect(x,y,w,h,fill='#FFFFFF',stroke='#687582',width=1.2,dashed=False,id=None):
 id=cell('',f'rounded=0;html=0;fillColor={fill};strokeColor={stroke};strokeWidth={width};dashed={int(dashed)};',x,y,w,h,id)
 dr.rectangle([round(x*scale),round(y*scale),round((x+w)*scale),round((y+h)*scale)],fill=fill,outline=stroke,width=round(width*scale))
 svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"'+(' stroke-dasharray="5 4"' if dashed else '')+'/>');return id

def box(id,lines,x,y,w,h,fill='#F1F5F8',stroke='#597187',size=18):
 rect(x,y,w,h,fill,stroke,id=id);text(lines,x+9,y+5,w-18,h-10,size=size);nodes[id]=(x,y,w,h);return id

def arrow(points,dashed=False):
 global counter
 counter+=1;e=SubElement(rt,'mxCell',id=f'e{counter}',style=f'edgeStyle=none;html=0;endArrow=block;endFill=1;strokeColor=#53616E;strokeWidth=1.4;rounded=0;dashed={int(dashed)};',edge='1',parent='1');g=SubElement(e,'mxGeometry',relative='1',**{'as':'geometry'});SubElement(g,'mxPoint',x=str(points[0][0]),y=str(points[0][1]),**{'as':'sourcePoint'});SubElement(g,'mxPoint',x=str(points[-1][0]),y=str(points[-1][1]),**{'as':'targetPoint'})
 if len(points)>2:
  a=SubElement(g,'Array',**{'as':'points'})
  for x,y in points[1:-1]:SubElement(a,'mxPoint',x=str(x),y=str(y))
 pts=[(round(x*scale),round(y*scale)) for x,y in points];dr.line(pts,fill='#53616E',width=round(1.4*scale))
 x,y=points[-1];px,py=points[-2];ang=math.atan2(y-py,x-px);tri=[(x,y),(x-8*math.cos(ang)+4*math.sin(ang),y-8*math.sin(ang)-4*math.cos(ang)),(x-8*math.cos(ang)-4*math.sin(ang),y-8*math.sin(ang)+4*math.cos(ang))];dr.polygon([(round(x*scale),round(y*scale)) for x,y in tri],fill='#53616E')
 svg.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in points)+'" fill="none" stroke="#53616E" stroke-width="1.4"'+(' stroke-dasharray="5 4"' if dashed else '')+'/>');svg.append('<polygon points="'+' '.join(f'{x},{y}' for x,y in tri)+'" fill="#53616E"/>')

def panel(letter,title,y,h):
 rect(10,y,980,h,fill='#FFFFFF',stroke='#C9D1D8',width=1)
 text(letter,28,y+12,26,27,21,True,align='left');text(title,65,y+12,900,27,19,True,align='left')

panel('A','Official data and frozen image representation',10,225)
box('data',['PatchCamelyon','Train 262,144','Validation 32,768','Test 32,768'],30,65,220,123)
box('preprocess',['96 × 96 RGB patches','Resize to 224 × 224','Normalize (ImageNet)','No augmentation'],285,65,190,123,size=17)
box('encoder',['Frozen ResNet50','ImageNet weights','Global average pooling','No fine tuning'],510,65,200,123,size=17)
box('cache',['2,048 features / patch','Shared frozen features','Saved sample and WSI IDs','Resumable feature cache'],745,65,220,123,size=16)
for a,b in [(250,285),(475,510),(710,745)]:arrow([(a,126),(b,126)])
text('The same encoder is used for all splits. Official test features are extracted only after the test lock.',30,200,935,25,size=16,align='left')
panel('B','Training only selection and shared feature transforms',252,198)
box('select',['Selection subset: 16,384 training patches','Three folds grouped by whole slide','Fit PCA and scaling within each fitting fold','Select shared neural rate and classical reference'],30,299,440,128,size=17)
box('pca',['Final transform fitted on all 262,144 training rows','PCA: 2,048 → 8 components, then standardize','z = standardized components, a = arctan(z)','Apply the fitted transform to held out data'],510,299,455,128,size=17)
arrow([(470,363),(510,363)])
panel('C','Four trainable heads and a selected classical reference',468,305)
arrow([(737,427),(737,518),(117,518),(117,533)])
for x in [307,497,687,877]:arrow([(737,518),(x,518),(x,533)])
box('mlp',['Standard MLP','Input z','8 → 32 → 1, tanh','321 parameters'],30,533,175,115,size=17)
box('atan',['Arctangent MLP','Input a','8 → 32 → 1, tanh','321 parameters'],220,533,175,115,size=17)
box('off',['RY, CNOT off','Input a, 8 wires','Two RY layers','25 parameters'],410,533,175,115,fill='#FFF4E9',stroke='#AE7844',size=17)
box('on',['RY, CNOT on','Input a, 8 wires','Two RY layers','25 parameters'],600,533,175,115,fill='#FFF4E9',stroke='#AE7844',size=17)
box('classic',['Classical reference','Input a','Fourier Nyström','256 landmarks + SGD'],790,533,175,115,fill='#F4F4F4',stroke='#797979',size=16)
box('schedule',['Neural heads: Adam, rate 0.003, batch 128, 100 epochs','Seeds 101, 202, 303. Matched CPU float64 training.','CNOT off/on share initialization, readout and batch order.'],30,673,575,78,size=17)
box('circuit',['RY(a) → two trainable RY layers','On: CNOT chain after each layer','8 Pauli Z means → linear readout','Exact simulation, no shots or hardware'],625,658,340,100,fill='#FFF9F2',stroke='#AE7844',size=16)
panel('D','Validation thresholds, artifact lock and official test evaluation',791,285)
box('validation',['Validation split by slide','17,090: threshold selection','15,678: diagnostic evaluation','Youden threshold per model'],30,843,285,128,size=18)
box('lock',['Lock fitted artifacts','Models, transforms, thresholds','Comparisons and analysis settings','SHA256 artifact checks'],355,843,285,128,size=17)
box('test',['Official test: 32,768 patches','129 whole slide groups','12 neural fits + 1 reference','Saved scores and probabilities'],680,843,285,128,size=17)
arrow([(315,907),(355,907)]);arrow([(640,907),(680,907)])
text(['Paired WSI bootstrap: 20,000 draws. Training seed SD reported separately.','Primary: CNOT on versus selected classical. Four secondary contrasts use adjusted intervals.'],30,996,935,58,size=17,align='left')
text('Post hoc checks: exact image overlap, slide influence, and circuit output and gradient validation.',30,1090,935,30,size=16,align='left')
arrow([(500,235),(500,252)])
arrow([(500,773),(500,791)])
svg.append('</svg>');(P/'Fig1_methodology.svg').write_text('\n'.join(svg));ElementTree(root).write(P/'Fig1_methodology.drawio',encoding='utf-8',xml_declaration=True)
im.save(P/'Fig1.tif',dpi=(400,400),compression='tiff_lzw');im.save(P/'Fig1.png',dpi=(400,400))
print('Created editable draw.io, SVG, PNG and 400 DPI TIFF.',im.size)
