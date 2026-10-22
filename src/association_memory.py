"""
association_memory.py
=====================

Identity Layer - 2. modul: **Uzun Sureli Hafiza ve Sahiplik Yonetimi**

Sistemdeki tum kalici kisi, esya ve sahiplik bilgileri burada saklanir.
Bu modul BoT-SORT Track ID'lerinden TAMAMEN BAGIMSIZ calisir.

--------------------------------------------------------------------
KALICI KIMLIK KURALLARI
--------------------------------------------------------------------
* Person UUID yalnizca bir kez uretilir ve ASLA degistirilmez.
* Track ID degisebilir ve gerektiginde guncellenir.
* Ayni kisi yeni bir Track ID ile tespit edilse bile mevcut UUID kullanilir.
* Tum kisi-esya iliskileri YALNIZCA UUID uzerinden yonetilir.
* Track ID hicbir zaman kalici kimlik veya sahiplik bilgisi degildir.
* Bir esya bir kez bir kisiye atandiktan sonra sahiplik degistirilmez.
--------------------------------------------------------------------

--------------------------------------------------------------------
SAHIPLIK: ZAMANSAL KANIT (try_assign_owner)
--------------------------------------------------------------------
Sahiplik degistirilemedigi icin atama aninin dogrulugu kritiktir. Esyayi
ilk goruldugu anda "en yakin kisiye" vermek uc tip hataya yol acar:
yanindan gecen birine, kimligi o an cozulemeyen gercek sahibin yerine
baska birine, perspektif nedeniyle onunden gecen birine.

Bu nedenle her esya icin, yakindaki her kisiye zamanla biriken ve
sonumlenen bir KANIT puani tutulur (PETS 2006/2007 calismalarindaki
"esyayi kim getirdi / kim yanindaydi" sorusunun geriye donuk cevabi):

    her karede : puan *= exp(-dt / OWNERSHIP_MEMORY_TIME)
                 yakin kisi icin puan += dt * agirlik(mesafe)
    atama      : lider puan >= OWNERSHIP_CONFIRM_TIME
                 VE lider >= OWNERSHIP_DOMINANCE * ikinci
                 VE liderin kimligi cozulmus

Yanindan gecen biri birkac onda bir saniyelik kanit toplar; esyanin
yaninda oturan veya onu tasiyarak getiren kisi saniyeler toplar.
Kimligi cozulmemis kisiler Track ID ile kanit biriktirir; UUID'leri
cozuldugunde kanit UUID'ye aktarilir.
--------------------------------------------------------------------

Icerdigi yapilar:
    PersonMemory | ItemMemory | AssociationTable | MemoryManager
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Deque, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from src.config import Config, load_config
from src.distance import (
    box_area,
    estimate_gap_m,
    euclidean_distance,
    get_box_center,
)
from src.utils import current_timestamp

logger = logging.getLogger(__name__)

PERSON_CLASS_NAME = "person"

BBox = Tuple[float, float, float, float]
Point = Tuple[float, float]


# =====================================================================
#  VERI YAPILARI
# =====================================================================


class ItemStatus(Enum):
    """Esyanin durum bilgisi."""

    NORMAL = "normal"        # Sahibinin yaninda
    LOST = "lost"            # Sahibi uzaklasti / esya gorunmuyor
    ABANDONED = "abandoned"  # Terk edilmis (uyari uretilir)

    @property
    def severity(self) -> int:
        """Siddet sirasi (NORMAL < LOST < ABANDONED). Durum makinesi geri gitmez."""
        return _STATUS_SEVERITY[self]


_STATUS_SEVERITY = {ItemStatus.NORMAL: 0, ItemStatus.LOST: 1, ItemStatus.ABANDONED: 2}


@dataclass(frozen=True)
class PersonCandidate:
    """Bu karede goruntudeki bir kisi (sahiplik adayi).

    `uuid` None ise kisi goruntude VAR fakat kimligi henuz cozulemedi
    (yeni Track, ReID kararsiz). Bu kisi sahip ATANAMAZ ama kanit biriktirir
    ve gercek sahip olabilecegi icin rakip olarak atamayi erteler.
    """

    track_id: int
    uuid: Optional[str]
    bbox: Tuple[float, float, float, float]

    @property
    def evidence_key(self) -> str:
        return self.uuid if self.uuid is not None else f"{_TRACK_KEY_PREFIX}{self.track_id}"


# Kimligi cozulmemis kisilerin kanit anahtari oneki.
_TRACK_KEY_PREFIX = "track:"
# Bu degerin altina sonumlenen kanit silinir (sozluk sismesin diye).
_MIN_EVIDENCE = 1e-3
# Esige tam mesafede duran adayin agirligi. Temas halinde 1.0'dir;
# boylece esit sureler icin YAKIN olan aday one gecer.
_EDGE_WEIGHT = 0.5


@dataclass
class _OwnershipEvidence:
    """Bir esya icin adaylarin biriken sahiplik kaniti."""

    last_update: float
    scores: Dict[str, float] = field(default_factory=dict)

    def ranked(self) -> List[Tuple[str, float]]:
        return sorted(self.scores.items(), key=lambda pair: pair[1], reverse=True)


@dataclass
class PersonMemory:
    """Bir kisinin kalici hafiza kaydi.

    `uuid` ASLA degismez. `track_id` gerektiginde guncellenir.
    """

    uuid: str
    max_embeddings: int
    track_id: Optional[int] = None
    first_seen: float = field(default_factory=current_timestamp)
    last_seen: float = field(default_factory=current_timestamp)
    last_bbox: Optional[BBox] = None
    last_center: Optional[Point] = None
    active: bool = True
    embeddings: Deque[np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        # FIFO: yeni embedding eklendiginde en eski otomatik dusar.
        self.embeddings = deque(maxlen=self.max_embeddings)

    def add_embedding(self, embedding: np.ndarray) -> None:
        """Embedding gecmisini FIFO mantigiyla gunceller."""
        if embedding is None:
            return
        self.embeddings.append(np.asarray(embedding, dtype=np.float32).ravel())

    @property
    def embedding_matrix(self) -> np.ndarray:
        """  Kişiye ait tüm kayıtlı embedding'leri matris (K, D) formatında döndürür.        """
        if not self.embeddings:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack(self.embeddings, axis=0)

    @property
    def mean_embedding(self) -> Optional[np.ndarray]:
        # Kayıtlı embedding'lerin ortalamasını hesaplayıp L2 normalize edilmiş olarak döndürür.
        # Kişiyi daha kararlı temsil etmek için kullanılır.
        matrix = self.embedding_matrix
        if matrix.size == 0:
            return None
        mean = matrix.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        return mean / norm if norm > 0 else mean

    def update_position(self, bbox: BBox, timestamp: Optional[float] = None) -> None:
        # Kişinin son görüldüğü bounding box, merkez noktası ve zaman bilgisini günceller. Ayrıca kişiyi aktif olarak işaretler.
        self.last_bbox = bbox
        self.last_center = get_box_center(bbox)
        self.last_seen = current_timestamp() if timestamp is None else timestamp
        self.active = True

    def to_dict(self) -> Dict:
        # Nesnedeki tüm bilgileri sözlük olarak döndürür. Loglama ve veri aktarma işlemlerinde kullanılır.
        return {
            "uuid": self.uuid,
            "track_id": self.track_id,
            "last_bbox": self.last_bbox,
            "last_center": self.last_center,
            "last_seen": self.last_seen,
            "active": self.active,
            "embedding_count": len(self.embeddings),
        }


@dataclass
class ItemMemory:
    """Bir esyanin kalici hafiza kaydi."""

    uuid: str
    class_id: int
    class_name: str
    track_id: Optional[int] = None
    owner_uuid: Optional[str] = None
    status: ItemStatus = ItemStatus.NORMAL
    first_seen: float = field(default_factory=current_timestamp)
    last_seen: float = field(default_factory=current_timestamp)
    last_bbox: Optional[BBox] = None
    last_center: Optional[Point] = None
    active: bool = True

    def update_position(self, bbox: BBox, timestamp: Optional[float] = None) -> None:
        # Eşyanın son görüldüğü bounding box, merkez noktası ve zaman bilgisini günceller. Ayrıca eşyayı aktif olarak işaretler.
        self.last_bbox = bbox
        self.last_center = get_box_center(bbox)
        self.last_seen = current_timestamp() if timestamp is None else timestamp
        self.active = True

    def to_dict(self) -> Dict:
        return {
            "uuid": self.uuid,
            "class_name": self.class_name,
            "track_id": self.track_id,
            "owner_uuid": self.owner_uuid,
            "status": self.status.value,
            "last_bbox": self.last_bbox,
            "last_center": self.last_center,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "active": self.active,
        }


class AssociationTable:
    # Bu sınıf, kişiler ile eşyalar arasındaki sahiplik ilişkisini yönetmek için kullanılır.
    # Hangi eşyanın hangi kişiye ait olduğunu saklar ve bir eşyanın aynı anda yalnızca tek bir sahibinin olmasını sağlar.
    """Kisi-esya sahiplik iliskisini yoneten tablo.

        Person UUID
                ├── Backpack UUID
                ├── Laptop UUID
                └── Cell Phone UUID

    * Bir kisinin birden fazla esyasi olabilir.
    * Bir esya ayni anda YALNIZCA tek bir kisiye ait olabilir.
    * Sahiplik bir kez kuruldugunda degistirilmez.
    """

    def __init__(self) -> None:
        self._person_to_items: Dict[str, Set[str]] = {}
        self._item_to_person: Dict[str, str] = {}

    def associate(self, person_uuid: str, item_uuid: str) -> bool:
        # Bir kişi ile eşya arasında sahiplik ilişkisi kurar. Eşyanın zaten bir sahibi varsa sahiplik değiştirilmez.
        """Sahiplik iliskisi kurar.

        Returns:
            Iliski kurulduysa True. Esya zaten bir sahibe baglanmissa
            False (mevcut sahiplik korunur).
        """
        current_owner = self._item_to_person.get(item_uuid)
        if current_owner is not None:
            if current_owner != person_uuid:
                logger.debug(
                    "Sahiplik degistirilmedi: %s zaten %s kisisine ait "
                    "(%s talebi reddedildi).",
                    item_uuid,
                    current_owner,
                    person_uuid,
                )
            return current_owner == person_uuid

        self._item_to_person[item_uuid] = person_uuid
        self._person_to_items.setdefault(person_uuid, set()).add(item_uuid)
        return True

    def owner_of(self, item_uuid: str) -> Optional[str]:
        # Belirtilen eşyanın sahibinin UUID'sini döndürür.
        return self._item_to_person.get(item_uuid)

    def items_of(self, person_uuid: str) -> Set[str]:
        # Belirtilen kişiye ait tüm eşya UUID'lerini döndürür.
        return set(self._person_to_items.get(person_uuid, ()))

    def has_items(self, person_uuid: str) -> bool:
        # Belirtilen kişinin en az bir eşyası olup olmadığını kontrol eder.
        return bool(self._person_to_items.get(person_uuid))

    def remove_item(self, item_uuid: str) -> None:
        # Bir eşyayı sahiplik tablosundan kaldırır ve sahibiyle olan ilişkiyi siler.
        owner = self._item_to_person.pop(item_uuid, None)
        if owner is not None and owner in self._person_to_items:
            self._person_to_items[owner].discard(item_uuid)
            if not self._person_to_items[owner]:
                del self._person_to_items[owner]

    def remove_person(self, person_uuid: str) -> None:
        # Bir kişiyi sahiplik tablosundan kaldırır ve ona ait tüm eşya ilişkilerini siler.
        for item_uuid in self._person_to_items.pop(person_uuid, set()):
            self._item_to_person.pop(item_uuid, None)

    def all_associations(self) -> Dict[str, Set[str]]:
        # Sistemde bulunan tüm kişi-eşya sahiplik ilişkilerini sözlük (dict) olarak döndürür.
        return {p: set(items) for p, items in self._person_to_items.items()}

    def __len__(self) -> int:
        return len(self._item_to_person)


# =====================================================================
#  MEMORY MANAGER
# =====================================================================

# kişi ve eşya hafızasını merkezi olarak yöneten katmandır.
# Kişileri ve eşyaları kalıcı UUID'leriyle saklar, sahiplik ilişkilerini yönetir, konum ve embedding bilgilerini
# günceller ve artık kullanılmayan kayıtları bellekten temizler.
class MemoryManager:
    """Sistemin uzun sureli hafizasini yoneten merkezi bilesen.

    Kullanim:
        memory = MemoryManager()
        person = memory.add_person(embedding=emb, track_id=7, bbox=box)
        item   = memory.register_item(...)
        memory.remove_expired_tracks()
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        proximity_fn: Optional[Callable[[BBox, BBox], float]] = None,
    ) -> None:
        self.config = config or load_config()

        # Geometrik hesaplama distance.py'den gelir; sahiplik KARARI bu
        # katmanindir. Test veya alternatif strateji (ornegin kalibre
        # edilmis zemin duzlemi) icin disaridan degistirilebilir.
        # Imza: (esya_kutusu, kisi_kutusu) -> YAKLASIK METRE.
        person_height_m = self.config.PERSON_HEIGHT_M
        self._proximity_m = proximity_fn or (
            lambda item_box, person_box: estimate_gap_m(item_box, person_box, person_height_m)
        )

        self.max_embeddings: int = self.config.MAX_EMBEDDING_HISTORY
        self.person_memory_size: int = self.config.PERSON_MEMORY_SIZE
        self.max_track_lost_time: float = self.config.MAX_TRACK_LOST_TIME
        self.memory_timeout: float = self.config.MEMORY_TIMEOUT
        self.purge_enabled: bool = self.config.MEMORY_PURGE_ENABLED
        self.ownership_distance_m: float = self.config.OWNERSHIP_DISTANCE_M
        self.ownership_confirm_time: float = self.config.OWNERSHIP_CONFIRM_TIME
        self.ownership_memory_time: float = self.config.OWNERSHIP_MEMORY_TIME
        self.ownership_dominance: float = self.config.OWNERSHIP_DOMINANCE
        # Kanit biriktirirken iki gozlem arasinda sayilabilecek en uzun sure.
        # Esya daha uzun sure gorunmediyse aradaki bosluk kanit sayilmaz.
        self.max_evidence_step: float = self.config.OCCLUSION_HOLD_TIME
        self.item_rematch_distance: float = self.config.ITEM_REMATCH_DISTANCE
        self.item_rematch_timeout: float = self.config.ITEM_REMATCH_TIMEOUT
        self.item_rematch_seen_gap: float = self.config.ITEM_REMATCH_SEEN_GAP
        self.item_rematch_max_area_ratio: float = self.config.ITEM_REMATCH_MAX_AREA_RATIO
        self.uuid_pad_width: int = self.config.UUID_PAD_WIDTH

        self._uuid_prefixes: Dict[str, str] = self.config.uuid_prefixes
        self._uuid_counters: Dict[str, int] = {}
        self._issued_uuids: Set[str] = set()  # Yeniden kullanimi engeller

        # Sahipligi henuz kurulmamis esyalar: Item UUID -> biriken kanit
        self._evidence: Dict[str, _OwnershipEvidence] = {}

        self.persons: Dict[str, PersonMemory] = {}
        self.items: Dict[str, ItemMemory] = {}
        self.associations = AssociationTable()

        logger.info(
            "MemoryManager hazir | embedding_gecmisi=%d (FIFO) | "
            "sahiplik: <=%.2f m, kanit>=%.1f sn, baskinlik x%.1f, sonum %.1f sn | "
            "rematch<=%.0fpx / %.0fs (gorulme araligi %.1fs, alan orani<=%.1f) | "
            "track_lost=%.0fs | timeout=%.0fs | purge=%s",
            self.max_embeddings,
            self.ownership_distance_m,
            self.ownership_confirm_time,
            self.ownership_dominance,
            self.ownership_memory_time,
            self.item_rematch_distance,
            self.item_rematch_timeout,
            self.item_rematch_seen_gap,
            self.item_rematch_max_area_ratio,
            self.max_track_lost_time,
            self.memory_timeout,
            self.purge_enabled,
        )

    # ----------------------------------------------------------------
    #  UUID URETIMI
    #  (utils.py'de degil burada olmalidir - bkz. Sistem Mimarisi - 7)
    # ----------------------------------------------------------------

    def _prefix_for(self, class_name: str) -> str:
        key = class_name.strip().lower().replace("_", " ")
        return self._uuid_prefixes.get(key, self._uuid_prefixes["_default"])

    def _next_uuid(self, prefix: str) -> str:
        """Benzersiz, okunabilir bir UUID uretir (P001, B003 ...).

        Uretilen hicbir UUID tekrar kullanilmaz.
        """
        while True:
            self._uuid_counters[prefix] = self._uuid_counters.get(prefix, 0) + 1
            candidate = f"{prefix}{self._uuid_counters[prefix]:0{self.uuid_pad_width}d}"
            if candidate not in self._issued_uuids:
                self._issued_uuids.add(candidate)
                return candidate

    def generate_person_uuid(self) -> str:
        """Yeni bir kalici Person UUID uretir."""
        return self._next_uuid(self._prefix_for(PERSON_CLASS_NAME))

    def generate_item_uuid(self, class_name: str) -> str:
        """Yeni bir kalici Item UUID uretir."""
        return self._next_uuid(self._prefix_for(class_name))

    # ----------------------------------------------------------------
    #  KAYIT OLUSTURMA
    # ----------------------------------------------------------------

    def add_person(
        self,
        uuid: Optional[str] = None,
        track_id: Optional[int] = None,
        bbox: Optional[BBox] = None,
        embedding: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ) -> PersonMemory:
        """Yeni bir kisi kaydi olusturur."""
        person_uuid = uuid or self.generate_person_uuid()
        if person_uuid in self.persons:
            return self.persons[person_uuid]

        now = current_timestamp() if timestamp is None else timestamp
        person = PersonMemory(
            uuid=person_uuid,
            max_embeddings=self.max_embeddings,
            track_id=track_id,
            first_seen=now,
            last_seen=now,
        )
        if bbox is not None:
            person.update_position(bbox, now)
        if embedding is not None:
            person.add_embedding(embedding)

        self.persons[person_uuid] = person
        self._enforce_person_capacity()
        logger.debug("Yeni kisi: %s (track=%s)", person_uuid, track_id)
        return person

    def add_item(
        self,
        class_id: int,
        class_name: str,
        track_id: Optional[int] = None,
        bbox: Optional[BBox] = None,
        owner_uuid: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> ItemMemory:
        """Yeni bir esya kaydi olusturur."""
        now = current_timestamp() if timestamp is None else timestamp
        item = ItemMemory(
            uuid=self.generate_item_uuid(class_name),
            class_id=class_id,
            class_name=class_name,
            track_id=track_id,
            first_seen=now,
            last_seen=now,
        )
        if bbox is not None:
            item.update_position(bbox, now)

        self.items[item.uuid] = item
        if owner_uuid is not None:
            self.associate(owner_uuid, item.uuid)
        logger.debug("Yeni esya: %s (%s, track=%s)", item.uuid, class_name, track_id)
        return item

    def associate(self, person_uuid: str, item_uuid: str) -> bool:
        """Kisi-esya sahiplik iliskisi kurar.

        Sahiplik bir kez kuruldugunda degistirilemez.
        """
        item = self.items.get(item_uuid)
        if item is None or person_uuid not in self.persons:
            logger.warning(
                "Iliski kurulamadi: bilinmeyen UUID (%s / %s)", person_uuid, item_uuid
            )
            return False

        created = self.associations.associate(person_uuid, item_uuid)
        if created:
            item.owner_uuid = self.associations.owner_of(item_uuid)
            self._evidence.pop(item_uuid, None)
        return created

    # ----------------------------------------------------------------
    #  ARAMA
    # ----------------------------------------------------------------

    def find_person(self, uuid: str) -> Optional[PersonMemory]:
        return self.persons.get(uuid)

    def find_item(self, uuid: str) -> Optional[ItemMemory]:
        return self.items.get(uuid)

    def find_person_by_track(self, track_id: int) -> Optional[PersonMemory]:
        """Track ID uzerinden kisi bulur.

        UYARI: Yalnizca aktif takip suresince gecerlidir. Kalici kimlik
        icin daima UUID kullanilmalidir.
        """
        for person in self.persons.values():
            if person.track_id == track_id and person.active:
                return person
        return None

    def find_item_by_track(self, track_id: int) -> Optional[ItemMemory]:
        for item in self.items.values():
            if item.track_id == track_id and item.active:
                return item
        return None

    def owner_of(self, item_uuid: str) -> Optional[str]:
        return self.associations.owner_of(item_uuid)

    def items_of(self, person_uuid: str) -> List[ItemMemory]:
        return [
            self.items[uuid]
            for uuid in self.associations.items_of(person_uuid)
            if uuid in self.items
        ]

    def recently_seen_items(
        self, now: float, max_age: float, exclude: Sequence[str] = ()
    ) -> List[ItemMemory]:
        """Bu karede gorunmeyen ama en fazla `max_age` sn once gorulmus,
        sahibi bilinen esyalar.

        Tespiti birkac kare kacan esyanin durumunun ve uyarisinin
        korunmasi icin kullanilir (karar `main.py`'dedir).
        """
        skip = set(exclude)
        return [
            item
            for item in self.items.values()
            if item.uuid not in skip
            and item.owner_uuid is not None
            and item.last_bbox is not None
            and 0.0 < now - item.last_seen <= max_age
        ]

    def provisional_owner(self, item_uuid: str) -> Optional[str]:
        """Sahipligi henuz kurulmamis esyanin su anki lider adayi (yalnizca
        gorsellestirme/tani icindir; KARAR DEGILDIR)."""
        evidence = self._evidence.get(item_uuid)
        if evidence is None or not evidence.scores:
            return None
        key, _ = evidence.ranked()[0]
        return None if key.startswith(_TRACK_KEY_PREFIX) else key

    def get_gallery(self) -> Dict[str, np.ndarray]:
        """ReID icin embedding galerisi: Person UUID -> (K, D)

        `reid.py` bu galeriyi yalnizca OKUR; degistirmez.
        """
        return {
            uuid: person.embedding_matrix
            for uuid, person in self.persons.items()
            if person.embeddings
        }

    # ----------------------------------------------------------------
    #  GUNCELLEME
    # ----------------------------------------------------------------

    def update_embedding(self, person_uuid: str, embedding: np.ndarray) -> bool:
        """Kisinin embedding gecmisini FIFO mantigiyla gunceller."""
        person = self.persons.get(person_uuid)
        if person is None:
            return False
        person.add_embedding(embedding)
        return True

    def update_position(
        self, uuid: str, bbox: BBox, timestamp: Optional[float] = None
    ) -> bool:
        """Kisi veya esyanin konumunu gunceller."""
        record = self.persons.get(uuid) or self.items.get(uuid)
        if record is None:
            return False
        record.update_position(bbox, timestamp)
        return True

    def bind_track(self, uuid: str, track_id: Optional[int]) -> bool:
        """Kayda guncel Track ID'yi baglar.

        Track ID degisimi UUID'yi ve sahiplik bilgisini ETKILEMEZ.
        """
        record = self.persons.get(uuid) or self.items.get(uuid)
        if record is None:
            return False
        record.track_id = track_id
        return True

    def observe_person(
        self,
        uuid: str,
        track_id: Optional[int] = None,
        bbox: Optional[BBox] = None,
        embedding: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ) -> Optional[PersonMemory]:
        """Mevcut bir kisiyi bu karede goruldugu sekilde gunceller."""
        person = self.persons.get(uuid)
        if person is None:
            return None
        person.track_id = track_id
        if bbox is not None:
            person.update_position(bbox, timestamp)
        else:
            person.last_seen = current_timestamp() if timestamp is None else timestamp
            person.active = True
        if embedding is not None:
            person.add_embedding(embedding)
        return person

    def set_status(self, item_uuid: str, status: ItemStatus) -> bool:
        item = self.items.get(item_uuid)
        if item is None:
            return False
        if item.status is not status:
            logger.debug("%s durumu: %s -> %s", item_uuid, item.status.value, status.value)
        item.status = status
        return True

    def mark_abandoned(self, item_uuid: str) -> bool:
        """Esyayi TERK EDILMIS olarak isaretler."""
        return self.set_status(item_uuid, ItemStatus.ABANDONED)

    def mark_lost(self, item_uuid: str) -> bool:
        return self.set_status(item_uuid, ItemStatus.LOST)

    def mark_normal(self, item_uuid: str) -> bool:
        return self.set_status(item_uuid, ItemStatus.NORMAL)

    # ----------------------------------------------------------------
    #  ILK ESLESTIRME KURALI
    # ----------------------------------------------------------------

    def register_item(
        self,
        class_id: int,
        class_name: str,
        track_id: int,
        bbox: BBox,
        person_candidates: Sequence[PersonCandidate] = (),
        timestamp: Optional[float] = None,
    ) -> ItemMemory:
        """Bir esyayi hafizaya kaydeder ve gerekiyorsa sahibini atar.

            1. Track ID zaten bir esyaya bagliysa o kayit guncellenir.
            2. Yeni Track ID ise ayni sinifta, ayni konumda ve BENZER
               BOYUTTA duran onceki bir kayit aranir (rematch).
            3. Hicbiri degilse yeni kayit olusturulur.

        Her uc durumda sahipsiz esya icin `try_assign_owner()` cagrilir;
        sahiplik zamanla biriken kanita gore atanir.

        Bu atama YALNIZCA BIR KEZ yapilir; sonrasinda sahiplik degismez.

        Args:
            person_candidates: Bu karede goruntudeki TUM kisiler
                (kimligi cozulemeyenler dahil, UUID=None).
        """
        now = current_timestamp() if timestamp is None else timestamp

        # 1) Bu Track ID zaten bir esyaya bagli mi?
        existing = self.find_item_by_track(track_id)
        if existing is not None:
            existing.update_position(bbox, now)
            self.try_assign_owner(existing, person_candidates, now)
            return existing

        # 2) Yeni Track ID: ayni sinifta, ayni konumda pasif bir kayit var mi?
        #    (Unutulan esya hareketsizdir; onunden biri gecince takip kopar
        #    ve BoT-SORT yeni bir Track ID uretir.)
        rematched = self._rematch_item(class_name, bbox, now)
        if rematched is not None:
            rematched.track_id = track_id
            rematched.update_position(bbox, now)
            logger.debug(
                "Esya yeniden eslesti: %s (yeni track=%s, sahip=%s korundu)",
                rematched.uuid,
                track_id,
                rematched.owner_uuid,
            )
            # Sahibi varsa dokunulmaz; sahipsizse kanit birikmeye devam eder.
            self.try_assign_owner(rematched, person_candidates, now)
            return rematched

        # 3) Gercekten ilk kez goruluyor.
        item = self.add_item(
            class_id=class_id,
            class_name=class_name,
            track_id=track_id,
            bbox=bbox,
            timestamp=now,
        )
        self.try_assign_owner(item, person_candidates, now)
        return item

    def try_assign_owner(
        self,
        item: ItemMemory,
        person_candidates: Sequence[PersonCandidate],
        timestamp: Optional[float] = None,
    ) -> bool:
        """Sahipsiz bir esya icin kanit biriktirir; yeterliyse sahibi atar.

        KURAL (ayrinti icin modul basligi)
        ----------------------------------
        * Esyaya OWNERSHIP_DISTANCE_M'den yakin her kisi, gecen sure ve
          yakinligi oraninda kanit kazanir; tum kanitlar zamanla sonumlenir.
        * Atama icin lider aday:
            - OWNERSHIP_CONFIRM_TIME saniyelik kanita ulasmali,
            - ikinci adayin en az OWNERSHIP_DOMINANCE kati kanita sahip olmali,
            - kimligi (UUID) cozulmus olmali.
        * Kosul saglanmazsa beklenir; emin olunamadiginda beklemek, geri
          alinamayan yanlis bir atamadan ucuzdur.

        Gerekce (cikti videolarinda gozlenen hatalar):
          * Esyanin uzerinden gecen yaya bir an "en yakin" olur; kanit
            toplayamaz.
          * Gercek sahibin Track ID'si degismis ve ReID onu henuz tanimamis
            olabilir; Track ID ile kanit toplamaya devam eder ve UUID'si
            cozulunce kaniti devralir.
          * Masayi paylasan iki kisi esit kanit toplar; baskinlik sarti
            saglanmadigi icin atama yapilmaz. Biri ayrilinca kalan kisi
            baskin hale gelir.
          * Esyayi getirip birakan ve hemen uzaklasan sahip, tasima
            sirasinda topladigi kanitla yine sahip olarak atanir.

        Sahibi olan esyalara DOKUNMAZ; mevcut sahiplik hicbir kosulda
        degistirilmez.

        Returns:
            Bu cagrida sahiplik kuruldu ise True.
        """
        if item.owner_uuid is not None:
            return False  # Sahiplik zaten kurulmus, degistirilemez.
        if item.last_bbox is None:
            return False

        now = current_timestamp() if timestamp is None else timestamp
        evidence = self._evidence.get(item.uuid)
        if evidence is None:
            evidence = self._evidence[item.uuid] = _OwnershipEvidence(last_update=now)

        # 1) Sonumleme. Esya uzun sure gorunmediyse aradaki sure kanit
        #    olarak sayilmaz ama sonumleme uygulanir.
        elapsed = max(0.0, now - evidence.last_update)
        step = elapsed if elapsed <= self.max_evidence_step else 0.0
        evidence.last_update = now
        if elapsed > 0.0:
            decay = math.exp(-elapsed / self.ownership_memory_time)
            for key in list(evidence.scores):
                evidence.scores[key] *= decay
                if evidence.scores[key] < _MIN_EVIDENCE:
                    del evidence.scores[key]

        # 2) Kimligi yeni cozulen kisinin Track ID ile topladigi kanit UUID'ye aktarilir.
        for candidate in person_candidates:
            if candidate.uuid is None:
                continue
            track_key = f"{_TRACK_KEY_PREFIX}{candidate.track_id}"
            if track_key in evidence.scores:
                evidence.scores[candidate.uuid] = (
                    evidence.scores.get(candidate.uuid, 0.0) + evidence.scores.pop(track_key)
                )

        # 3) Bu karedeki yakin kisiler kanit kazanir.
        if step > 0.0:
            for candidate in person_candidates:
                gap_m = self._proximity_m(item.last_bbox, candidate.bbox)
                if gap_m > self.ownership_distance_m:
                    continue
                weight = 1.0 - _EDGE_WEIGHT * (gap_m / self.ownership_distance_m)
                key = candidate.evidence_key
                evidence.scores[key] = evidence.scores.get(key, 0.0) + weight * step

        # 4) Karar.
        ranked = evidence.ranked()
        if not ranked:
            return False
        leader, leader_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0

        if leader_score < self.ownership_confirm_time:
            return False
        if runner_up > 0.0 and leader_score < self.ownership_dominance * runner_up:
            logger.debug(
                "%s icin sahiplik belirsiz: %s=%.2f, ikinci=%.2f (baskinlik x%.1f gerekli)",
                item.uuid, leader, leader_score, runner_up, self.ownership_dominance,
            )
            return False
        if leader.startswith(_TRACK_KEY_PREFIX) or leader not in self.persons:
            return False  # Lider aday henuz tanimlanmadi: beklenir.

        assigned = self.associate(leader, item.uuid)
        if assigned:
            logger.debug(
                "%s sahibi atandi: %s (kanit %.2f sn, ikinci %.2f)",
                item.uuid, leader, leader_score, runner_up,
            )
        return assigned

    def is_owner_pending(self, item_uuid: str) -> bool:
        """Esya icin sahiplik kaniti birikiyor mu? (tani/gorsellestirme)"""
        evidence = self._evidence.get(item_uuid)
        return evidence is not None and bool(evidence.scores)

    def _rematch_item(
        self, class_name: str, bbox: BBox, now: float
    ) -> Optional[ItemMemory]:
        """Ayni sinifta, ayni konumda, benzer boyutta duran onceki bir kaydi bulur."""
        if self.item_rematch_distance <= 0:
            return None

        center = get_box_center(bbox)
        area = box_area(bbox)
        best_item, best_distance = None, self.item_rematch_distance
        for item in self.items.values():
            if item.class_name != class_name or item.last_center is None:
                continue
            # Bu karede (veya cok kisa sure once) gorulmus bir esya baska bir
            # takibe baglidir; ona dokunulmaz.
            #
            # Olcut `active` bayragi DEGILDIR: o bayrak ancak
            # MAX_TRACK_LOST_TIME (30 sn) sonunda duser. Takibi 2 saniye once
            # kopmus bir esya hala active=True ve track_id dolu gorunur, bu da
            # rematch'in atlanmasina, yeni bir UUID uretilmesine ve sahipligin
            # o anda en yakinda duran kisiye sessizce gecmesine yol acar.
            if (
                item.track_id is not None
                and now - item.last_seen <= self.item_rematch_seen_gap
            ):
                continue
            if now - item.last_seen > self.item_rematch_timeout:
                continue
            # Boyut kontrolu: ayni noktada beliren, boyutu cok farkli bir
            # tespit (ornegin elde tutulan kagidin "handbag" sanilmasi) ayni
            # esya degildir; eski kaydin sahibini ve durumunu devralmamali.
            if item.last_bbox is not None and not self._similar_size(
                area, box_area(item.last_bbox)
            ):
                continue
            distance = euclidean_distance(center, item.last_center)
            if distance <= best_distance:
                best_item, best_distance = item, distance
        return best_item

    def _similar_size(self, area_a: float, area_b: float) -> bool:
        if area_a <= 0.0 or area_b <= 0.0:
            return False
        return max(area_a, area_b) / min(area_a, area_b) <= self.item_rematch_max_area_ratio

    # ----------------------------------------------------------------
    #  BELLEK BAKIMI
    # ----------------------------------------------------------------

    def remove_expired_tracks(self, timestamp: Optional[float] = None) -> Dict[str, int]:
        """Suresi dolan Track ID baglantilarini dusurur.

        Iki asamali calisir:
          1. MAX_TRACK_LOST_TIME asilirsa: kayit PASIFLESIR ve Track ID
             baglantisi kopar. UUID ve embedding'ler KORUNUR; boylece
             kisi geri geldiginde ReID ayni UUID'yi bulabilir.
          2. MEMORY_TIMEOUT asilirsa (ve purge aciksa): kayit silinebilir.
             Esyasi olan kisiler ve terk edilmis esyalar ASLA silinmez.

        Returns:
            {'deactivated': n, 'purged': n}
        """
        now = current_timestamp() if timestamp is None else timestamp
        deactivated = 0

        for record in list(self.persons.values()) + list(self.items.values()):
            if record.active and now - record.last_seen > self.max_track_lost_time:
                record.active = False
                record.track_id = None
                deactivated += 1

        purged = self._purge(now) if self.purge_enabled else 0
        if deactivated or purged:
            logger.debug(
                "Bellek bakimi: %d pasiflestirildi, %d silindi", deactivated, purged
            )
        return {"deactivated": deactivated, "purged": purged}

    def _purge(self, now: float) -> int:
        """Suresi dolmus kayitlari guvenli sekilde siler.

        Sahiplik butunlugu her seyin onundedir: sahibi olan bir esya
        silinirse, tekrar goruldugunde yeni bir UUID alir ve o anda en
        yakinda duran KISIYE atanir. Bu, sahipligin sessizce el
        degistirmesi demektir. Bu nedenle sahipli esyalar silinmez.
        """
        purged = 0

        for uuid, item in list(self.items.items()):
            if item.active or now - item.last_seen <= self.memory_timeout:
                continue
            if item.status is ItemStatus.ABANDONED:
                continue  # Terk edilmis esya asla silinmez
            if self.associations.owner_of(uuid) is not None:
                continue  # Sahipli esya silinmez (sahiplik korunur)
            del self.items[uuid]
            self._evidence.pop(uuid, None)
            purged += 1

        for uuid, person in list(self.persons.items()):
            if person.active or now - person.last_seen <= self.memory_timeout:
                continue
            if self.associations.has_items(uuid):
                continue  # Esyasi olan kisi silinmez (sahiplik korunur)
            del self.persons[uuid]
            purged += 1

        return purged

    def _enforce_person_capacity(self) -> None:
        """PERSON_MEMORY_SIZE asildiginda en eski uygun kaydi dusurur."""
        while len(self.persons) > self.person_memory_size:
            candidates = [
                person
                for person in self.persons.values()
                if not person.active and not self.associations.has_items(person.uuid)
            ]
            if not candidates:
                logger.warning(
                    "PERSON_MEMORY_SIZE (%d) asildi fakat silinebilecek kayit yok.",
                    self.person_memory_size,
                )
                return
            oldest = min(candidates, key=lambda p: p.last_seen)
            del self.persons[oldest.uuid]
            logger.debug("Kapasite asimi: %s bellekten dusuruldu.", oldest.uuid)

    # ----------------------------------------------------------------
    #  DURUM BILGISI
    # ----------------------------------------------------------------

    def active_persons(self) -> List[PersonMemory]:
        return [p for p in self.persons.values() if p.active]

    def active_items(self) -> List[ItemMemory]:
        return [i for i in self.items.values() if i.active]

    def abandoned_items(self) -> List[ItemMemory]:
        return [i for i in self.items.values() if i.status is ItemStatus.ABANDONED]

    def stats(self) -> Dict[str, int]:
        """HUD ve loglama icin ozet bilgi."""
        return {
            "persons_total": len(self.persons),
            "persons_active": len(self.active_persons()),
            "items_total": len(self.items),
            "items_active": len(self.active_items()),
            "items_abandoned": len(self.abandoned_items()),
            "associations": len(self.associations),
        }

    def __repr__(self) -> str:
        stats = self.stats()
        return (
            f"MemoryManager(persons={stats['persons_total']}, "
            f"items={stats['items_total']}, "
            f"associations={stats['associations']})"
        )
