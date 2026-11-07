"""Monocarpic (hapaxanthic) plants: the flowering rosette or plant dies after its single bloom.

Owner's rule (Ben, 2026-11-06): "plants that die after blooming should not be shown in bloom." The image pass
never gives one a flower stalk, spike or inflorescence, they never count toward the bloom share, and an added
stalk is a defect (`flower_stalk_added`).

The curated list is the 3D skill's (pacifica-planting-renders scripts/monocarpic.py); keep the two in step.
Annuals and biennials grown for their flowers are NOT covered. A plant record's explicit `monocarpic: true|false`
overrides the list.
"""
import re

REASON = 'monocarpic: not shown in bloom'

GENERA = ('Agave', 'Mangave', 'Furcraea', 'Puya', 'Hesperoyucca', 'Sempervivum', 'Aeonium',
          'Aechmea', 'Guzmania', 'Neoregelia', 'Vriesea', 'Tillandsia')
# Species in mixed genera (binomial -> monocarpic). These win over the genus default.
SPECIES = {'Echium wildpretii': True, 'Echium pininana': True, 'Echium candicans': False,
           'Yucca whipplei': True,
           'Kalanchoe luciae': True, 'Kalanchoe thyrsiflora': True, 'Kalanchoe daigremontiana': True,
           'Kalanchoe delagoensis': True, 'Kalanchoe tubiflora': True}
HYBRID = {'x', '×'}


def genus_species(taxon):
    """(Genus, 'Genus epithet' or None) from a scheduled name; cultivars, ranks and hybrid markers ignored."""
    words = [w.strip('.,;:()[]') for w in re.sub(r"['‘’\"“”][^'‘’\"“”]*['‘’\"“”]", ' ', str(taxon)).split()]
    words = [w for w in words if w]
    if words and words[0].casefold() in HYBRID:
        words = words[1:]
    if not words:
        return None, None
    genus = words[0].lstrip('×').capitalize()
    rest = [w for w in words[1:] if w.casefold() not in HYBRID]
    epithet = rest[0].lstrip('×') if rest else ''
    if not re.fullmatch(r'[a-z][a-z-]*[a-z]', epithet) or epithet in ('sp', 'spp'):
        return genus, None
    return genus, genus + ' ' + epithet


def is_monocarpic(taxon, override=None):
    if override is not None:
        if not isinstance(override, bool):
            raise ValueError('monocarpic flag must be true or false')
        return override
    genus, binomial = genus_species(taxon)
    if binomial in SPECIES:
        return SPECIES[binomial]
    return genus in GENERA
