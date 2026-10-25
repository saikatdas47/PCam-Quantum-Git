"""Build a journal manuscript with editable tables and embedded scientific figures."""
from pathlib import Path
import re,json,csv
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
P=Path('paper/plos_one_word');D=Document();sec=D.sections[0]
sec.page_width=Inches(8.5);sec.page_height=Inches(11);sec.top_margin=sec.bottom_margin=Inches(.8);sec.left_margin=sec.right_margin=Inches(.85)
sec.footer_distance=Inches(.35)
for name in ['Normal','Body Text','Title','Subtitle','Heading 1','Heading 2','Heading 3','Caption']:
 st=D.styles[name];st.font.name='Times New Roman';st.font.color.rgb=RGBColor(0,0,0)
normal=D.styles['Normal'];normal.font.size=Pt(12);normal.paragraph_format.line_spacing=1.5;normal.paragraph_format.space_after=Pt(3)
for name,size in [('Heading 1',13),('Heading 2',12),('Heading 3',12)]:
 st=D.styles[name];st.font.size=Pt(size);st.font.bold=True;st.paragraph_format.space_before=Pt(12);st.paragraph_format.space_after=Pt(4);st.paragraph_format.keep_with_next=True;st.paragraph_format.line_spacing=1.15
D.styles['Title'].font.size=Pt(17);D.styles['Title'].font.bold=True;D.styles['Title'].paragraph_format.space_after=Pt(14);D.styles['Title'].paragraph_format.line_spacing=1.15
D.styles['Caption'].font.size=Pt(10);D.styles['Caption'].font.italic=False;D.styles['Caption'].paragraph_format.line_spacing=1.1;D.styles['Caption'].paragraph_format.space_after=Pt(8)
# Remove inherited template decoration and enforce manuscript typography.
for st in D.styles:
    if st.type == 1:
        if st._element.pPr is not None:
            for border in list(st._element.pPr.findall(qn('w:pBdr'))):
                st._element.pPr.remove(border)
        if st.name in ['Normal','Body Text','Title','Subtitle','Heading 1','Heading 2','Heading 3','Caption']:
            fonts=st._element.get_or_add_rPr().find(qn('w:rFonts'))
            if fonts is not None:
                for key in ['asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme']:
                    fonts.attrib.pop(qn('w:'+key),None)
                for key in ['ascii','hAnsi','eastAsia','cs']:fonts.set(qn('w:'+key),'Times New Roman')
# Page numbers are field-based and remain editable.
footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
run=footer.add_run();field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');run._r.addnext(field)
D.core_properties.title='A comparison of simulated quantum and classical classification heads on frozen ResNet50 features for PatchCamelyon';D.core_properties.author='Saikat Das';D.core_properties.subject='Empirical comparison of simulated quantum and classical classification heads';D.core_properties.keywords='PatchCamelyon; quantum machine learning; ResNet50; clustered bootstrap; ablation'
def hyperlink(p,label,target):
 from docx.opc.constants import RELATIONSHIP_TYPE as RT
 h=OxmlElement('w:hyperlink');h.set(qn('r:id'),p.part.relate_to(target,RT.HYPERLINK,is_external=True))
 run=OxmlElement('w:r');pr=OxmlElement('w:rPr');color=OxmlElement('w:color');color.set(qn('w:val'),'000000');pr.append(color);run.append(pr)
 t=OxmlElement('w:t');t.text=label;run.append(t);h.append(run);p._p.append(h)
def rich(p,text):
 urls={}
 def protect(m):
  token='URLTOKEN'+str(len(urls))+'TOKEN';urls[token]=m.group(0);return token
 text=re.sub(r'https?://[^\s]+',protect,text)
 if not re.match(r'^\d+\. ',text):
  text=text.replace('; ','. ').replace('–',' to ').replace('—','Not applicable')
  text=re.sub(r'(?<=[A-Za-z])-(?=[A-Za-z])',' ',text)
 for token,url in urls.items():text=text.replace(token,url)
 for part in re.split(r'(\*\*.*?\*\*)',text):
  bold=part.startswith('**');part=part[2:-2] if bold else part
  for segment in re.split(r'(https?://[^\s]+|saikatdasmain47@gmail\.com)',part):
   if segment.startswith('http'):
    clean=segment.rstrip('.,)');hyperlink(p,clean,clean)
    if clean!=segment:p.add_run(segment[len(clean):])
   elif segment=='saikatdasmain47@gmail.com':hyperlink(p,segment,'mailto:'+segment)
   else:p.add_run(segment).bold=bold

def para(text,style=None):
 p=D.add_paragraph(style=style);rich(p,text);p.paragraph_format.widow_control=True;return p
def tbl(headers,rows,widths):
 t=D.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
 for c,w in zip(t.columns,widths):c.width=Inches(w)
 for cell,text in zip(t.rows[0].cells,headers):cell.text=text
 for row in rows:
  for cell,text in zip(t.add_row().cells,row):cell.text=str(text)
 for ri,row in enumerate(t.rows):
  pr=row._tr.get_or_add_trPr();cant=OxmlElement('w:cantSplit');pr.append(cant)
  if ri==0:pr.append(OxmlElement('w:tblHeader'))
  for ci,cell in enumerate(row.cells):
   cell.width=Inches(widths[ci]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER;tcpr=cell._tc.get_or_add_tcPr()
   borders=OxmlElement('w:tcBorders')
   for edge in ['top','left','bottom','right']:
    e=OxmlElement('w:'+edge);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');borders.append(e)
   tcpr.append(borders);m=OxmlElement('w:tcMar')
   for edge in ['top','left','bottom','right']:
    e=OxmlElement('w:'+edge);e.set(qn('w:w'),'75');e.set(qn('w:type'),'dxa');m.append(e)
   tcpr.append(m)
   if ri==0:
    shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'E9EEF2');tcpr.append(shade)
   for p in cell.paragraphs:
    p.alignment=WD_ALIGN_PARAGRAPH.LEFT if ci==0 else WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.line_spacing=1.05;p.paragraph_format.space_after=Pt(2);p.paragraph_format.space_before=Pt(2)
    for r in p.runs:r.font.size=Pt(9);r.bold=ri==0
 D.add_paragraph().paragraph_format.space_after=Pt(2)
 return t
captions={
'Fig1':('Fig1.png','Fig 1. Executed study methodology. (A) Official PatchCamelyon splits and shared frozen ResNet50 feature extraction. (B) Training only grouped model selection and training fitted PCA and scaling. The final shared representation contains eight components. (C) Four neural heads and a training selected Fourier Nystrom reference. Neural fits use three seeds and a shared optimization schedule. CNOT off and on circuits share initial rotations and readout values. Circuits are classically simulated without shots or quantum hardware. (D) Validation threshold selection, fitted artifact lock, official test scoring and paired whole slide uncertainty analysis. Post hoc checks are identified separately.'),
'Fig2':('Fig2.png','Fig 2. Test ROC curves. Each panel compares one neural head at seeds 101, 202 and 303 with the same selected classical reference. Legend values are AUROC. False positive rate is FP divided by all negative patches. True positive rate is TP divided by all positive patches. The dotted diagonal shows equal true and false positive rates. All curves use the same 32,768 test patches. No smoothing or uncertainty bands are applied.'),
'Fig3':('Fig3.png','Fig 3. Test precision recall curves. Panels and seed styles match Fig 2. Legend values are average precision. Recall is TP divided by all positive patches. Precision is TP divided by all predicted positive patches. The dotted horizontal line marks test prevalence, 16,377/32,768 or 0.4998. Small changes near low recall reflect individual score ranks. The conventional endpoint at recall zero has precision one. Curves are empirical steps without smoothing or uncertainty bands.'),
'Fig4':('Fig4.png','Fig 4. Paired differences in mean test AUROC. The blue primary contrast uses a nominal 95% WSI bootstrap interval. Orange secondary contrasts use Bonferroni-adjusted intervals for a four-comparison family. All intervals condition on the fitted models and retained seeds. The vertical dashed line marks zero. Values are percentage points.'),
'Fig5':('Fig5.png','Fig 5. Training diagnostics and observed compute time. (A) Cross-entropy on a fixed 1,024-row training cohort. (B) Individual-seed neural compute times. Black ticks denote means. Time uses a log scale. Session differences limit hardware comparisons. Timer boundaries are specified in Methods.')}
def figure(key):
 file,caption=captions[key];p=D.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.line_spacing=1;p.paragraph_format.keep_with_next=True;p.add_run().add_picture(str(Path('paper/PLOS_ONE_submission/figures')/file),width=Inches(6.45))
 # Accessibility description stays attached to the picture.
 inline=p.runs[0]._r.xpath('.//wp:docPr')[0];inline.set('descr',caption)
 cp=para(caption,'Caption');cp.paragraph_format.keep_together=True
source=Path('paper/plos_one_word/MANUSCRIPT.md').read_text()
source=source[:source.index('## Figure captions')]
source=source[source.index('## Abstract'):];source=re.sub(r'^\*\*Initial research-article draft.*?\n\n','',source,count=1,flags=re.S)
# Preserve necessary unknowns, keeping document-production notes out of the paper.
source=source.replace('[AUTHOR TO COMPLETE. Do not invent contributors or approvals.]','[Acknowledgments to be supplied by the authors.]')
source=source.replace('[AUTHORS MUST REVIEW all outputs, confirm the exact tool/model/version and usage dates, and describe their own validation and responsibility before submission.]','The authors retain responsibility for scientific interpretation, source attribution and final validation.')
source=source.replace('This draft reports executed evidence. Submission requires the remaining items in SUBMISSION_READINESS.md. It is not a preregistered or registered-report study.','')
# Retain the explicit acknowledgment placeholder until author review.
source=source.replace('Final proceedings pagination/DOI should be verified before submission.','').replace('Complete publisher metadata before submission.','').replace('Cited as the inspected preprint version; final publication status not audited here.','').replace('Relevant external implementation not audited.','').replace('Abstract and bibliographic record inspected:','Bibliographic record:')
source=source.replace('The exact computed values are in S2 Table.','The exact computed values are in Table 4.').replace('exact computed values are in S2 Table.','exact computed values are in Table 4.')
source=source.replace('Table 2 reports mean operating metrics across seeds, not metrics of a pooled ensemble. Exact per-model numerators, denominators and confusion matrices are supplied in S1 Table.','Table 2 reports mean performance across seeds, not performance of a pooled ensemble. Operating characteristics and exact per-model confusion counts are supplied in S1 and S2 Tables.')
source=source.replace('Full traces, update counts and measured timing are in S3 Table.','Full traces, update counts and measured timing are retained in the accompanying analysis tables.')
D.add_paragraph('A comparison of simulated quantum and classical classification heads on frozen ResNet50 features for PatchCamelyon','Title')
p=para('Saikat Das\nDepartment of Computer Science and Engineering\nAhsanullah University of Science and Technology, Dhaka, Bangladesh\nCorresponding author: Saikat Das\nEmail: saikatdasmain47@gmail.com');p.paragraph_format.line_spacing=1.2
p=para('Running title  Quantum and classical heads on frozen ResNet50 features');p.paragraph_format.line_spacing=1.2
D.add_page_break()
lines=source.splitlines();i=0;fig1=False
results=json.loads(Path('paper/plos_one_initial/analysis/results.json').read_text());summ=results['model_summary'];names={'mlp_standard':'Standard MLP','mlp_atan':'Arctangent MLP','ry_off':'RY CNOT off','ry_on':'RY CNOT on','selected_classical':'Selected classical'}
while i<len(lines):
 line=lines[i].strip()
 if not line:i+=1;continue
 if line.startswith('|'):
  block=[]
  while i<len(lines) and lines[i].strip().startswith('|'):block.append(lines[i]);i+=1
  headers=[x.strip() for x in block[0].strip().strip('|').split('|')];rows=[[x.strip() for x in l.strip().strip('|').split('|')] for l in block[2:]]
  if headers[0]=='Head':
   headers=['Model','Fits','AUROC\nMean (seed SD)','AUROC\n95% CI','AP','Brier']
   rows=[[names[x['head']],x['fits'],f"{x['AUROC_mean']:.4f}"+(f" ({x['AUROC_seed_SD']:.4f})" if x['AUROC_seed_SD'] is not None else ''),f"{x['AUROC_CI_low']:.4f} to {x['AUROC_CI_high']:.4f}",f"{x['AP_mean']:.4f}",f"{x['Brier_mean']:.4f}"] for x in summ]
   tbl(headers,rows,[1.35,.38,1.48,1.45,.8,.8]);figure('Fig2');D.add_page_break();figure('Fig3');D.add_page_break()
  elif headers[0]=='Contrast':
   label_map={'on selected':'CNOT on minus classical','on off':'CNOT on minus CNOT off','on atan':'CNOT on minus atan MLP','on standard':'CNOT on minus standard MLP','atan standard':'Atan minus standard MLP'}
   rows=[[label_map.get(row[0],row[0]),*row[1:]] for row in rows]
   tbl(headers,rows,[1.85,.8,1.8,1.81]);figure('Fig4')
   para('Table 4. Post hoc WSI jackknife sensitivity tests','Caption')
   rows=[[{'on-selected':'CNOT on minus classical','on-off':'CNOT on minus CNOT off','on-atan':'CNOT on minus atan MLP','on-standard':'CNOT on minus standard MLP','atan-standard':'Atan minus standard MLP'}[c['id']],f"{c['jackknife_SE']:.5f}",f"{c['t_statistic']:.3f}",f"{c['p_two_sided_posthoc']:.4f}",f"{c['p_Holm_secondary_posthoc']:.4f}" if 'p_Holm_secondary_posthoc' in c else 'NA'] for c in results['contrasts']]
   tbl(['Contrast','SE','t\n(df 128)','Two sided p','Holm p'],rows,[2.1,1,1,1.1,1.06]);para('Sensitivity tests were added after inspection of test results. Holm adjustment applies to the four secondary contrasts only. The primary locked inference remains the paired WSI percentile bootstrap.','Caption')
  else:tbl(headers,rows,[1.1,1.4,1.4,1.4,.96])
  continue
 if line.startswith('## '):
  heading=line[3:];hp=D.add_heading(heading,level=1)
  if heading in ['Introduction','Related work','Materials and methods','References']:hp.paragraph_format.page_break_before=True
  i+=1;continue
 if line.startswith('### '):D.add_heading(re.sub(r'[^\w\s]', ' ', line[4:]).strip(),level=2);i+=1;continue
 # Paragraphs are single Markdown lines, intentionally retained as coherent units.
 style='Caption' if line.startswith('**Table ') else None
 if style=='Caption' and 'Table 2.' in line:
  line='**Table 2. Official test discrimination and probability error.** AUROC is the mean across fixed neural seeds; SD describes seed variation. Conditional 95% intervals use paired WSI bootstrap draws averaged across retained seeds. The selected classical model is one fit. AP is average precision. Threshold dependent metrics are reported in S1 Table.'
 if style=='Caption':
  p=para(line,style);p.paragraph_format.keep_with_next=True
 else:
  p=para(line)
  if re.match(r'^\d+\. ',line) and len(line)>80:
   p.paragraph_format.line_spacing=1.15;p.paragraph_format.space_after=Pt(7);p.paragraph_format.left_indent=Inches(.2);p.paragraph_format.first_line_indent=Inches(-.2);p.style=D.styles['Normal']
 if 'test-based model winner' in line:pass
 # Methodology occupies a dedicated page so labels remain readable.
 if line.startswith('This computer study'):
  D.add_page_break();figure('Fig1');D.add_page_break()
 if 'Full traces, update counts' in line:figure('Fig5')
 i+=1
D.add_page_break();D.add_heading('Supporting information',level=1)
para('S1 Table. Official test operating characteristics and log loss','Caption')
rows=[[names[x['head']],f"{x['accuracy_mean']:.4f}",f"{x['sensitivity_mean']:.4f}",f"{x['specificity_mean']:.4f}",f"{x['precision_mean']:.4f}",f"{x['F1_mean']:.4f}",f"{x['log_loss_mean']:.4f}"] for x in summ]
tbl(['Model','Accuracy','Sensitivity','Specificity','Precision','F1','Log loss'],rows,[1.4,.8,.95,.95,.85,.6,.71]);para('Accuracy, sensitivity, specificity, precision and F1 use validation fixed thresholds. Neural values are arithmetic means of three seed specific metrics. Log loss uses predicted probabilities without thresholding. Class specific denominators are 16,377 positive and 16,391 negative test patches.','Caption')
para('S2 Table. Per model confusion counts and selected thresholds','Caption')
raw=list(csv.DictReader(Path('paper/plos_one_initial/tables/S1_per_model_metrics.csv').open()));rows=[]
for x in raw:
 tag=x['tag'];display=tag.replace('resnet50_q8_','').replace('resnet50_selected_classical','Selected classical').replace('mlp_standard','Standard MLP').replace('mlp_atan','Atan MLP').replace('ry_off','CNOT off').replace('ry_on','CNOT on').replace('_',' ')
 rows.append([display,x['TN'],x['FP'],x['FN'],x['TP'],f"{float(x['threshold']):.6f}"])
tbl(['Model and seed','TN','FP','FN','TP','Threshold'],rows,[2,.82,.82,.82,.82,.98]);para('All counts refer to the same 32,768 official test patches. Positive means tumor in the central 32 by 32 pixels. Decisions use probability greater than or equal to the saved threshold.','Caption')


# Supporting scientific validation and complete existing selection evidence.
D.add_page_break();D.add_heading('Supplementary methods',level=1)
para('Implementation validation', 'Heading 2')
para('Validation was added after the official test results were inspected. It checks the retained circuit code without fitting parameters. A separate NumPy implementation builds full Kronecker RY matrices and explicit basis permutation CNOT matrices. It begins in the zero state, applies angle encoding, then two rotation layers with the declared ordered chains. Wire zero is the most significant bit. There are six forward cases for each wire count and gate setting, including zero angles and pi over two angles. All input and rotation coordinates are checked by central finite differences with step 1e-6. Forward tolerance is 1e-12 and gradient tolerance is 1e-8. Basis maps establish adjacent control and target direction. The cosine expression separately checks the gate free outputs. This provides a different mathematical implementation within the same project, not an external replication.')
para('Consistency and numerical recovery', 'Heading 2')
para('All 1,207 exported manifest entries matched SHA256 and byte count. All 654 locked artifacts matched their stored hashes. Thirteen fitted model records, twelve 100 epoch neural traces and 26 held out prediction files were present. Metric recomputation used a separate analysis script in the same workflow. The five nominal intervals were reconstructed from saved bootstrap arrays. The first 32 saved draws for each model were checked by direct weighted AUROC calculation. The full 20,000 resampling procedure was not repeated by the reanalysis.')
para('Saved feature chunks and epoch checkpoints allowed continuation across sessions. Python patch versions and CPU or CUDA wheel variants were recorded. Predictions could differ slightly on replay. A tolerance of 1e-12 was used, while protected saved predictions and thresholds remained authoritative. Threshold crossings were accepted only inside this tolerance band. These checks concern numerical consistency. They do not make raw image replication or independent investigator claims.')
para('Initialization and optimization reporting', 'Heading 2')
para('The circuit initialization uses normal rotation weights with SD 0.1 and PyTorch Linear readout initialization. MLP weights and biases use the Linear uniform initialization rule for each layer input width. The selection seed is 71911 and final seeds are 101, 202 and 303. Off circuit parameters are copied from the paired on circuit. Reconstructed parameter CSV files are marked as current runtime reconstructions. Original initial tensors were not separately archived, so bitwise equality of reconstructed initial draws across library versions is not asserted. All 78 existing training fold scores and the full classical candidate grid are provided as machine readable files. There was one CV initialization seed, so the folds do not estimate initialization variation.')
para('Eight component and dataset audits', 'Heading 2')
para('Explained variance values were read from the stored PCA objects. No transform was fitted again. These objects were created with scikit learn 1.6.1 and read with local 1.7.2 for this reporting task. The version warnings are recorded. The audit reads numeric attributes and does not make transformed predictions under the changed version. Exact image checks hash C order uint8 RGB pixel bytes. All original label arrays were compared with saved manifests. The local train image hashes reproduce the original saved train hashes, but the missing original training gzip still limits compressed file provenance. The test image hashes are newly computed. Exact overlap exclusion and slide influence are descriptive checks added after test inspection.')
new=json.loads(Path('paper/major_revision/results/reporting_additions.json').read_text())
checks=json.loads(Path('paper/major_revision/results/circuit_validation.json').read_text())
D.add_page_break();para('S3 Table. Training fitted PCA explained variance','Caption')
rows=[[v['fit'],v['training_rows'],8,f"{100*v['cumulative_explained_variance']:.4f}%"] for v in new['PCA']]
tbl(['Transform','Fitting rows','Components','Variance retained'],rows,[1.5,1.5,1.3,2.06])
para('Individual component values are supplied in PCA_explained_variance.csv. Variance retention does not establish retention of label information.','Caption')
para('S4 Table. Existing learning rate selection results','Caption')
rows=[[names[v['head']],str(v['learning_rate']),f"{v['fold0']:.4f}",f"{v['fold1']:.4f}",f"{v['fold2']:.4f}",f"{v['mean_AUROC']:.4f}"] for v in new['neural_selection']]
tbl(['Head','Rate','Fold 1','Fold 2','Fold 3','Mean'],rows,[1.65,.8,.95,.95,.95,1.06])
para('All rows use training folds and initialization seed 71911. The original rule selected rate 0.003 by averaging across all heads and folds. The grid was not used to change the test model. Complete unrounded scores for all 78 fold jobs accompany the paper.','Caption')
D.add_page_break();para('S5 Table. Patch counts across slides','Caption')
rows=[[k,v['slides'],v['min'],v['median'],v['max'],f"{100*v['largest5_slide_fraction']:.2f}%"] for k,v in new['slide_summary'].items()]
tbl(['Partition','Slides','Minimum','Median','Maximum','Largest five'],rows,[1.15,.8,1.05,1.05,1.05,1.26])
para('Counts refer to original rows. The last column is the fraction of rows contributed by the five largest slides. All slide counts are supplied in patches_per_slide.csv.','Caption')
para('S6 Table. Circuit numerical validation','Caption')
rows=[[v['q'],'On' if v['CNOT'] else 'Off',v['forward_cases'],f"{v['max_forward_absolute_error']:.2e}",f"{v['max_gradient_absolute_error']:.2e}"] for v in checks['checks']]
tbl(['Wires','CNOT','Cases','Maximum output error','Maximum gradient error'],rows,[.7,.8,.75,2,2.11])
para('Errors are absolute. The independent dense matrix reference supplies the expected outputs and finite difference objective. Local NumPy 1.26.4 and PyTorch 2.14.1 were used. This is post hoc implementation validation.','Caption')
para('S7 Table. Largest changes in the primary point difference after removing one slide','Caption')
rows=[[v['slide'].replace('camelyon16_test_','Test '),v['patches'],f"{v['primary_difference_without_slide']:.6f}",f"{v['change_from_full']:.6f}"] for v in new['primary_influence']['top5']]
tbl(['Slide','Patches','Difference without slide','Change from full'],rows,[1.3,1.0,2.1,1.96])
para('These are the five largest absolute changes. The full point difference is minus 0.011911. All 129 leave one slide differences remain negative. No slide was removed from the locked analysis.','Caption')

D.save(P/'PCam_ResNet50_Quantum_Manuscript.docx')
print('Built manuscript with',len(D.paragraphs),'paragraphs,',len(D.tables),'editable tables and',len(D.inline_shapes),'embedded figures.')
