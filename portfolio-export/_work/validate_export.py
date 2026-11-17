"""Validate a strict portfolio whitelist, then create and reopen one final ZIP."""
from pathlib import Path, PurePosixPath
from html.parser import HTMLParser
from urllib.parse import urlsplit, parse_qsl, unquote
from collections import Counter
import argparse
import ast
import io
import json
import os
import re
import stat
import subprocess
import sys
import zipfile

sys.dont_write_bytecode=True
from build_export import ROOT, BASE, WORK, DEST, SLUG, DATE, COMMIT, sha, pair, git


def load(rel):
    return json.loads((DEST/rel).read_text(encoding='utf8'))


def assert_safe_rel(rel):
    p=PurePosixPath(rel)
    assert rel and not p.is_absolute() and '..' not in p.parts and '\\' not in rel and ':' not in rel,rel
    assert not any(v in {'.git','.env','.venv','venv','node_modules','__pycache__'} for v in p.parts),rel
    assert p.suffix.lower() not in {'.zip','.pt','.engine','.onnx','.bin','.npz','.mp4','.webm','.avi','.ipynb','.sql','.csv','.jpg','.png','.gif'},rel


def secret_and_link_review(entries):
    patterns={
        'private-key':re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
        'github-token':re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{24,}|github_pat_[A-Za-z0-9_]{35,})\b'),
        'openai-like-key':re.compile(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}\b'),
        'aws-key':re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
        'jwt':re.compile(r'\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b'),
        'credential-assignment':re.compile(r'''(?i)\b(?:password|passwd|api[_-]?key|client[_-]?secret|access[_-]?token)\s*[:=]\s*["'](?!<|\$\{|\{)[^"'\r\n]{8,}["']'''),
        'email':re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
        'personal-path':re.compile(r'(?i)(?:[a-z]:[/\\]Users[/\\]|/home/[^\s/]+|/Users/[^\s/]+)'),
        'legacy-private-path':re.compile(r'(?i)(?:D:[/\\]ML[/\\]|density-aware-yolo|lozy-warehouse)'),
        'phone-id':re.compile(r'(?<![A-Za-z0-9.])(?:\+62|0062|08[1-9])[\d .-]{7,16}(?![A-Za-z0-9])'),
        'image-data-uri':re.compile(r'data:image/[^;]+;base64,',re.I),
    }
    hits=[]; urls=set(); text_count=0
    allowed_hosts={'github.com','api.github.com','download.pytorch.org','fonts.gstatic.com','fonts.googleapis.com','www.w3.org'}
    for rel,data in entries.items():
        if rel.endswith('.webp'): continue
        txt=data.decode('utf8');text_count+=1
        assert '\ufffd' not in txt,('encoding-replacement-character',rel)
        assert not any(c in txt for c in ['\u202a','\u202b','\u202d','\u202e','\u2066','\u2067','\u2068']),('bidi-control',rel)
        for label,pat in patterns.items():
            for match in pat.finditer(txt):
                hits.append({'file':rel,'category':label,'line':txt.count('\n',0,match.start())+1})
        for u in re.findall(r'''https?://[^\s<>`"'\)\]]+''',txt):
            u=u.rstrip('.,;')
            if '{a.bind}' in u and rel=='source-code/tools/serve_reports.py':
                continue
            parts=urlsplit(u)
            assert not parts.username and not parts.password,('credential-bearing-url',rel)
            assert parts.hostname in allowed_hosts,('unreviewed-url-host',rel,parts.hostname)
            assert not any(re.search(r'(?i)token|secret|signature|credential|password|api[_-]?key',k) for k,v in parse_qsl(parts.query)),('token-query',rel)
            urls.add(u)
    assert not hits, json.dumps(hits,ensure_ascii=True)
    return {'textFilesScanned':text_count,'patternFindings':[],'distinctReviewedUrls':len(urls),
            'reviewedUrlHosts':sorted({urlsplit(u).hostname for u in urls}),
            'method':pair('Full text scan plus targeted manual inspection; all final images also reviewed visually. Pattern matching alone is not proof of image safety.',
                          'Pemindaian teks penuh serta inspeksi manual terarah; seluruh gambar akhir juga ditinjau visual. Pencocokan pola saja bukan bukti keamanan gambar.')}


def paired_text_review(obj):
    count=0
    def walk(x):
        nonlocal count
        if isinstance(x,dict):
            if 'en' in x or 'id' in x and set(x).issubset({'en','id'}):
                if 'en' in x:
                    assert 'id' in x and isinstance(x['en'],str) and isinstance(x['id'],str) and x['en'].strip() and x['id'].strip()
                    en_numbers=re.findall(r'\d+',x['en']);id_numbers=re.findall(r'\d+',x['id'])
                    assert en_numbers==id_numbers,('bilingual-numeric-mismatch',x['en'],x['id'])
                    count+=1
            for v in x.values(): walk(v)
        elif isinstance(x,list):
            for v in x: walk(v)
    walk(obj)
    return count


def preflight(final=False):
    from PIL import Image,ImageChops
    import yaml
    project=load('project.json');assets=load('assets-manifest.json');source=load('source-manifest.json')
    assert project['publicationMode']=='public' and project['defaultLanguage']=='en'
    required={'slug','title','domain','summary','stack','problem','personalContribution','features','engineeringFocus','architecture','implementation','technicalDecisions','proof','limitations','sources','coverAsset','recommendedWebsiteAssets'}
    assert required<=set(project)
    assert project['slug']==SLUG and project['demo'] is None
    assert 4<=len(project['stack'])<=8
    for lang in ('en','id'): assert 25<=len(project['summary'][lang].split())<=45
    assert project['snapshot']['commit']==COMMIT
    assert project['licenseStatus']['repositoryLicenseSpdx'] is None
    root_files={'README.md','project.json','case-study.en.md','case-study.id.md','assets-manifest.json','source-manifest.json',
                'technical/architecture.md','technical/development.md','technical/api-and-data.md','technical/third-party-notices.md',
                'technical/environment-observed.json','technical/aggregate-evidence.json',
                'technical/person-counter-example.md','technical/person-counter-example-evidence.json',
                'internal/evidence.md','internal/missing-information.md','internal/publication-review.md'}
    expected=set(root_files)
    assert source['fileCount']==27 and len(source['files'])==27
    assert source['pythonModuleCount']==16 and source['historicalHtmlSourceCount']==4
    tracked=set(git('ls-files').splitlines())
    py_count=html_count=yaml_count=0
    class HTMLCheck(HTMLParser): pass
    for rec in source['files']:
        rel=rec['path'];expected.add(rel);p=DEST/rel;data=p.read_bytes()
        assert len(data)==rec['bytes'] and sha(data)==rec['sha256'],rel
        if rec['repositoryPath']:
            assert rec['repositoryPath'] in tracked
            original=(ROOT/rec['repositoryPath']).read_bytes()
            assert sha(original)==rec['originalSha256']
            if not rec['redacted']: assert original==data
        if p.suffix=='.py':
            ast.parse(data.decode('utf8'),filename=rel);py_count+=1
        elif p.suffix=='.html': HTMLCheck().feed(data.decode('utf8'));html_count+=1
        elif p.suffix=='.yaml':
            cfg=yaml.safe_load(data);assert cfg['path']=='<DATASET_ROOT>' and cfg['nc']==1 and cfg['names']==['person'];yaml_count+=1
    assert (py_count,html_count,yaml_count)==(16,4,2)
    ids={a['id'] for a in assets['assets']}
    assert len(ids)==7 and project['coverAsset'] in ids
    assert project['recommendedWebsiteAssets']==['asset-01','asset-07','asset-02','asset-05']==assets['recommendedWebsiteAssets']
    pixels_preserved=0
    for a in assets['assets']:
        for record in [a,a['preview']]:
            rel=record['path'];expected.add(rel);path=DEST/rel
            assert path.name==path.name.lower() and re.fullmatch(r'\d{2}-[a-z0-9-]+(?:-preview)?\.webp',path.name)
            data=path.read_bytes();assert record['bytes']==len(data) and record['sha256']==sha(data)
            with Image.open(path) as im:
                im.load();assert im.size==(record['width'],record['height']) and im.format=='WEBP'
                assert not any(k.lower() in {'exif','xmp','icc_profile','comment','description','gps'} for k in im.info),(rel,im.info.keys())
        assert a['preview']['width']<=800 and a['preview']['width']<=a['width'] and a['preview']['height']<=a['height']
        assert abs(a['width']/a['height']-a['preview']['width']/a['preview']['height'])<.02
        if a['provenance']['kind']=='existing-project-chart':
            with Image.open(ROOT/a['provenance']['repositoryPath']) as orig, Image.open(DEST/a['path']) as out:
                assert out.size==orig.size and ImageChops.difference(orig.convert('RGB'),out.convert('RGB')).getbbox() is None
            pixels_preserved+=1
        if final: assert a['review']['status']=='reviewed' and a['review']['fullSizeVisualReview'] and a['review']['previewVisualReview']
    cover=next(a for a in assets['assets'] if a['id']==project['coverAsset']);assert (cover['width'],cover['height'])==(1600,1000)
    for rel in expected: assert_safe_rel(rel)
    actual={p.relative_to(DEST).as_posix() for p in DEST.rglob('*') if p.is_file()}
    assert actual-{'internal/validation-report.json'}==expected,('unexpected-or-missing-files',sorted(actual-expected-{'internal/validation-report.json'}),sorted(expected-actual))
    assert not any(p.is_symlink() for p in DEST.rglob('*'))
    entries={rel:(DEST/rel).read_bytes() for rel in sorted(expected)}
    json_count=paired_count=0
    for rel,data in entries.items():
        if rel.endswith('.json'):
            obj=json.loads(data);json_count+=1;paired_count+=paired_text_review(obj)
    broken_links=[]
    for rel,data in entries.items():
        if not rel.endswith('.md') or rel.startswith('source-code/') and rel!='source-code/EXPORT-NOTES.md': continue
        text=data.decode('utf8')
        for target in re.findall(r'\]\(([^\)]+)\)',text):
            if target.startswith(('https://','http://','#')): continue
            target=unquote(target.split('#')[0].split(' "')[0].strip('<>'))
            path=(DEST/rel).parent/target
            if not path.exists(): broken_links.append({'from':rel,'target':target})
    assert not broken_links,broken_links
    evidence_ids=set(re.findall(r'\| (E\d\d) \|',(DEST/'internal/evidence.md').read_text(encoding='utf8')))
    def evidence_refs(obj):
        if isinstance(obj,dict):
            if 'evidenceIds' in obj: assert set(obj['evidenceIds'])<=evidence_ids
            if 'assetIds' in obj: assert set(obj['assetIds'])<=ids
            if 'packagePath' in obj: assert obj['packagePath'] in entries
            for v in obj.values(): evidence_refs(v)
        elif isinstance(obj,list):
            for v in obj: evidence_refs(v)
    evidence_refs(project)
    for lang in ('en','id'):
        text=(DEST/f'case-study.{lang}.md').read_text(encoding='utf8')
        assert len(re.findall(r'^## \d+\.',text,re.M))==10
    aggregate=load('technical/aggregate-evidence.json')
    assert aggregate['dataset']['images']==4746 and aggregate['dataset']['totalBoxes']==18445
    assert aggregate['verification']['newTrainingRun'] is False and aggregate['verification']['newVideoInferenceRun'] is False
    assert aggregate['verification']['newSingleFrameInferenceRun'] is True
    assert len(aggregate['evaluation'])==7
    assert all(x['maxAbsoluteRecomputeDelta']==0 for x in aggregate['evaluation'].values())
    assert len(aggregate['modelIntegrity']['checksumChecks'])==4
    assert aggregate['focusedChecks']['sharedInvalidId']['outCount']==1
    assert aggregate['focusedChecks']['storeTimeBlockRounding']['validationBlocks']==2
    scan=secret_and_link_review(entries)
    originals=json.loads((WORK/'original-hashes.json').read_text())
    assert all(sha((ROOT/rel).read_bytes())==digest for rel,digest in originals.items()),'Original tracked file changed'
    assert git('status','--porcelain','--untracked-files=no')=='' and git('rev-parse','HEAD')==COMMIT
    result={'status':'passed','checkedOn':DATE,'timezone':'Asia/Jakarta','sourceFiles':27,'pythonModulesParsed':py_count,
            'htmlSourcesParsed':html_count,'yamlConfigurationsParsed':yaml_count,'jsonFilesParsed':json_count,
            'localizedTextPairsChecked':paired_count,'mainVisuals':7,'previewVisuals':7,'unchangedChartPixelComparisons':pixels_preserved,
            'brokenDocumentationLinks':[],'intentionallyOmittedLegacyHtmlReferences':len(source['omittedHtmlReferences']),
            'privacy':scan,'originalTrackedFilesUnchanged':len(originals),'originalTrackedWorkingTreeClean':True,
            'sourceSnapshotCommit':COMMIT,'newTrainingRun':False,'newVideoInferenceRun':False,'newSingleFrameInferenceRun':True,'repositoryAutomatedTestSuiteRun':False,
            'codeCoverageMeasured':False,'cachedPredictionOutputsRecomputed':7,'modelChecksumEntriesVerified':4,
            'focusedSyntheticAndSourceChecks':['shared-invalid-tracker-id','installed-auto-optimizer','store-time-block-rounding'],
            'visualReview': 'completed-full-size-main-and-preview' if final else 'awaiting-final-main-and-preview-attestation',
            'distributionScope':pair('Reviewed custom-code and documentation handoff with aggregate charts and one privacy-redacted frame included at the user\'s explicit direction; full dataset, raw media and models are excluded. The upstream image license remains undefined.',
                                     'Handoff kode custom dan dokumentasi yang ditinjau dengan grafik agregat serta satu frame dengan redaksi privasi yang disertakan atas instruksi eksplisit pengguna; dataset lengkap, media mentah, dan model dikeluarkan. Lisensi gambar upstream tetap belum ditentukan.')}
    return entries,result


def cli_checks():
    cases=[('tools/train.py','--list'),('tools/evaluate.py','--help'),('tools/count_people.py','--help')]
    results=[]
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['YOLO_AUTOINSTALL']='False'
    for rel,flag in cases:
        run=subprocess.run([sys.executable,'-B',str(DEST/'source-code'/rel),flag],cwd=DEST/'source-code',env=env,
                           capture_output=True,text=True,encoding='utf8',errors='replace',timeout=60)
        assert run.returncode==0,(rel,flag,'nonzero exit')
        assert run.stdout.strip(),(rel,'empty help/list output')
        results.append({'sourcePath':'source-code/'+rel,'argument':flag,'exitCode':run.returncode,
                        'scope':'CLI help/list only; no inference or training'})
    return results


def package(entries,report):
    final_path=BASE/f'portfolio-public-{SLUG}-{DATE}.zip'
    if final_path.exists():
        previous=json.loads((WORK/'delivery.json').read_text(encoding='utf8'))
        assert sha(final_path.read_bytes())==previous['sha256'],'Refusing to replace a ZIP that differs from the previous verified delivery'
    report['cliChecks']=cli_checks()
    report['archiveValidation']={'status':'passed','singleRoot':SLUG,'entriesMatchReviewedWhitelist':True,
                                 'entryBytesMatchReviewedFiles':True,'crcCheckPassed':True,'nestedArchives':0,'symlinks':0,
                                 'unsafePaths':0,'duplicateEntries':0,'validationReportIncluded':True}
    report['totalPackageFiles']=len(entries)+1
    report['jsonFilesParsedBeforeArchive']=report['jsonFilesParsed']
    report['jsonFilesParsed']+=1
    report['privacy']['textFilesScanned']+=1
    report_bytes=(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf8')
    final_entries=dict(entries);final_entries['internal/validation-report.json']=report_bytes
    assert not secret_and_link_review(final_entries)['patternFindings']
    mem=io.BytesIO()
    with zipfile.ZipFile(mem,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for rel,data in sorted(final_entries.items()):
            assert_safe_rel(rel)
            info=zipfile.ZipInfo(SLUG+'/'+rel,date_time=(2026,11,16,12,0,0))
            info.create_system=3;info.external_attr=(stat.S_IFREG|0o644)<<16
            info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,data,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    payload=mem.getvalue()
    def reopen_and_check(blob):
        with zipfile.ZipFile(blob,'r') as archive:
            names=archive.namelist();expected=[SLUG+'/'+r for r in sorted(final_entries)]
            assert names==expected and len(names)==len(set(n.casefold() for n in names))
            assert archive.testzip() is None
            for info in archive.infolist():
                parts=PurePosixPath(info.filename).parts
                assert parts[0]==SLUG and '..' not in parts and not PurePosixPath(info.filename).is_absolute()
                assert stat.S_ISREG(info.external_attr>>16) and not stat.S_ISLNK(info.external_attr>>16)
                rel='/'.join(parts[1:]);data=archive.read(info.filename)
                assert data==final_entries[rel]
                if rel.endswith('.json'): json.loads(data)
                if rel.endswith('.webp'):
                    from PIL import Image
                    with Image.open(io.BytesIO(data)) as im: im.load()
    reopen_and_check(io.BytesIO(payload))
    # The released report describes checks already passed on these exact archive bytes.
    (DEST/'internal/validation-report.json').write_bytes(report_bytes)
    temporary=final_path.with_suffix('.zip.tmp')
    assert not temporary.exists(),'Refusing to replace an existing temporary archive'
    temporary.write_bytes(payload)
    reopen_and_check(temporary)
    os.replace(temporary,final_path)
    reopen_and_check(final_path)
    assert git('status','--porcelain','--untracked-files=no')==''
    delivery={'zipPath':str(final_path),'bytes':len(payload),'sha256':sha(payload),'packageFiles':len(final_entries),
              'sourceFiles':27,'pythonModules':16,'mainVisuals':7,'previewVisuals':7,'validation':'passed',
              'recommendedWebsiteAssets':['asset-01','asset-07','asset-02','asset-05']}
    (WORK/'delivery.json').write_text(json.dumps(delivery,indent=2),encoding='utf8')
    print(json.dumps(delivery,ensure_ascii=True))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--finalize',action='store_true');a=ap.parse_args()
    entries,report=preflight(final=a.finalize)
    if a.finalize: package(entries,report)
    else:
        (WORK/'preflight.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
        print(json.dumps(report,ensure_ascii=True))
