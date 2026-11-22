"""Archive and configured-weight validation without downloads or neural mocks."""
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import subprocess

from disc_morphology.backend import safe_extract,config_for,COMMIT
from disc_morphology.dicom import sha256


class ArchiveTests(unittest.TestCase):
    def archive(self,root,name,kind=None):
        p=root/'archive.tar.gz'
        with tarfile.open(p,'w:gz') as t:
            info=tarfile.TarInfo(name)
            if kind:
                info.type=kind;info.linkname='/outside'
                t.addfile(info)
            else:
                body=b'content';info.size=len(body);t.addfile(info,io.BytesIO(body))
        return p

    def test_only_supported_regular_assets_are_extracted(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);archive=self.archive(r,'Dataset1/trainer/plans.json')
            safe_extract(archive,r/'out','Dataset1')
            self.assertEqual((r/'out/Dataset1/trainer/plans.json').read_bytes(),b'content')

    def test_traversal_and_wrong_dataset_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp)
            for name in ('Dataset1/../../outside.pth','/outside.pth','Other/model.pth'):
                with self.assertRaisesRegex(ValueError,'Unsafe'):
                    safe_extract(self.archive(r,name),r/'out','Dataset1')

    def test_archive_links_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp)
            for kind in (tarfile.SYMTYPE,tarfile.LNKTYPE):
                with self.assertRaisesRegex(ValueError,'Unsafe'):
                    safe_extract(self.archive(r,'Dataset1/redirect.pth',kind),r/'out','Dataset1')

    def test_model_config_checks_converted_weight_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);python=r/'python';python.write_text('test runtime')
            weight=r/'safe_data'/'weight.pth';weight.parent.mkdir();weight.write_bytes(b'known')
            (r/'config.json').write_text(json.dumps({'source_commit':COMMIT,'python':str(python),'source':str(r/'source')}))
            (r/'checkpoint-inventory.json').write_text(json.dumps([{'relative_path':'safe_data/weight.pth','sha256':sha256(weight)}]))
            completed=subprocess.CompletedProcess([],0,stdout=COMMIT+'\n')
            with patch('disc_morphology.backend._run',return_value=completed):
                self.assertEqual(config_for(r)['source_commit'],COMMIT)
                weight.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'checkpoint changed'):config_for(r)


if __name__=='__main__':unittest.main()
