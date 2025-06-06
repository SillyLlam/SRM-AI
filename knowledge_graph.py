from typing import Dict, List, Any
import networkx as nx
from datetime import datetime

class SRMKnowledgeGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.entities = {
            'campuses': {
                'Kattankulathur': {
                    'location': 'Chennai',
                    'established': '1985',
                    'address': 'SRM Nagar, Kattankulathur, Chengalpattu District, Tamil Nadu - 603203',
                    'landmarks': ['Tech Park', 'University Building', 'Central Library']
                },
                'Delhi-NCR': {
                    'location': 'Sonepat',
                    'established': '2013',
                    'address': 'Delhi-NCR Campus Plot No. 39, Rajiv Gandhi Education City, PS Rai, Sonepat, Haryana - 131029'
                },
                'Amaravati': {
                    'location': 'Andhra Pradesh',
                    'established': '2017',
                    'address': 'Neerukonda, Mangalagiri Mandal, Guntur District, Andhra Pradesh - 522502'
                },
                'Sikkim': {
                    'location': 'Gangtok',
                    'established': '2019',
                    'address': '5th Mile, Tadong, Gangtok, East Sikkim - 737102'
                }
            },
            'locations': {
                'Tech Park': {
                    'description': 'A state-of-the-art facility housing research labs and industry collaboration centers',
                    'location': 'Kattankulathur Campus',
                    'facilities': ['Research Labs', 'Innovation Center', 'Industry Collaboration Space'],
                    'map_link': 'https://maps.app.goo.gl/HvLKqGK8TFE5QWLP6'
                },
                'Central Library': {
                    'description': 'Multi-story library with vast collection of books, journals, and digital resources',
                    'location': 'Kattankulathur Campus',
                    'facilities': ['Reading Halls', 'Digital Library', 'Conference Rooms'],
                    'map_link': 'https://maps.app.goo.gl/HvLKqGK8TFE5QWLP6'
                },
                'University Building': {
                    'description': 'Main administrative building housing key offices and departments',
                    'location': 'Kattankulathur Campus',
                    'facilities': ['Administrative Offices', 'Admission Office', 'Exam Cell']
                }
            },
            'programs': {
                'Engineering': {
                    'degrees': ['B.Tech', 'M.Tech', 'Ph.D'],
                    'departments': [
                        'Computer Science',
                        'Mechanical',
                        'Civil',
                        'Electronics and Communication',
                        'Electrical and Electronics'
                    ]
                },
                'Medicine': {
                    'degrees': ['MBBS', 'MD', 'MS'],
                    'departments': [
                        'General Medicine',
                        'Surgery',
                        'Pediatrics',
                        'Orthopedics'
                    ]
                },
                'Management': {
                    'degrees': ['BBA', 'MBA', 'Ph.D'],
                    'departments': [
                        'Finance',
                        'Marketing',
                        'Human Resources',
                        'Operations'
                    ]
                },
                'Law': {
                    'degrees': ['BBA LLB', 'LLM'],
                    'departments': [
                        'Corporate Law',
                        'Criminal Law',
                        'Civil Law'
                    ]
                }
            },
            'facilities': {
                'hostels': {
                    'types': ['Men\'s Hostel', 'Women\'s Hostel'],
                    'amenities': ['Wi-Fi', 'Gym', 'Reading Room', 'Cafeteria']
                },
                'sports': {
                    'indoor': ['Badminton', 'Table Tennis', 'Chess'],
                    'outdoor': ['Cricket', 'Football', 'Basketball']
                },
                'transportation': {
                    'services': ['College Bus', 'Shuttle Service'],
                    'routes': ['Chennai City', 'Local Areas']
                }
            }
        }
        self._build_graph()
    
    def _build_graph(self):
        """Builds the initial knowledge graph structure."""
        # Add campus nodes
        for campus, details in self.entities['campuses'].items():
            self.graph.add_node(campus, type='campus', **details)
        
        # Add location nodes and relationships
        for location, details in self.entities['locations'].items():
            self.graph.add_node(location, type='location', **details)
            campus = details['location']
            self.graph.add_edge(campus, location, relationship='has_location')
        
        # Add program nodes and relationships
        for program, details in self.entities['programs'].items():
            self.graph.add_node(program, type='program', **details)
            for campus in self.entities['campuses']:
                self.graph.add_edge(campus, program, relationship='offers')
                for degree in details['degrees']:
                    self.graph.add_node(f"{program}_{degree}", type='degree',
                                      program=program, degree=degree)
                    self.graph.add_edge(program, f"{program}_{degree}",
                                      relationship='has_degree')
        
        # Add facility nodes
        for facility_type, details in self.entities['facilities'].items():
            self.graph.add_node(facility_type, type='facility', **details)
            for campus in self.entities['campuses']:
                self.graph.add_edge(campus, facility_type, relationship='has_facility')
    
    def query(self, entity_type: str, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Query the knowledge graph for entities matching the given type and filters.
        
        Args:
            entity_type: Type of entity to query (campus, location, program, facility)
            filters: Dictionary of attribute-value pairs to filter results
            
        Returns:
            List of matching entities with their attributes
        """
        results = []
        for node in self.graph.nodes(data=True):
            node_id, attrs = node
            
            # Skip if node type doesn't match
            if 'type' not in attrs or attrs['type'] != entity_type:
                continue
                
            # Check if node matches all filters
            if filters:
                matches_filters = all(
                    key in attrs and attrs[key] == value 
                    for key, value in filters.items()
                )
                if not matches_filters:
                    continue
            
            # Add matching node to results
            results.append({'id': node_id, **attrs})
        
        return results
    
    def get_related_entities(self, entity_id: str, relationship_type: str = None) -> List[Dict[str, Any]]:
        """Get entities related to the given entity."""
        related = []
        for _, neighbor, edge_data in self.graph.edges(entity_id, data=True):
            if relationship_type and edge_data.get('relationship') != relationship_type:
                continue
            neighbor_data = self.graph.nodes[neighbor]
            related.append({
                'id': neighbor,
                'relationship': edge_data.get('relationship'),
                **neighbor_data
            })
        return related

    def add_entity(self, entity_id: str, entity_type: str, attributes: Dict[str, Any] = None):
        """Add a new entity to the knowledge graph."""
        self.graph.add_node(entity_id, type=entity_type, **(attributes or {}))
    
    def add_relationship(self, from_entity: str, to_entity: str, relationship_type: str):
        """Add a relationship between two entities."""
        self.graph.add_edge(from_entity, to_entity, relationship=relationship_type)
        
    def search_by_text(self, query: str) -> List[Dict[str, Any]]:
        """Search for entities by matching text in their attributes."""
        results = []
        query = query.lower()
        
        for node in self.graph.nodes(data=True):
            node_id, attrs = node
            
            # Check node ID
            if query in node_id.lower():
                results.append({'id': node_id, **attrs})
                continue
            
            # Check attribute values
            for key, value in attrs.items():
                if isinstance(value, str) and query in value.lower():
                    results.append({'id': node_id, **attrs})
                    break
                elif isinstance(value, list) and any(query in str(v).lower() for v in value):
                    results.append({'id': node_id, **attrs})
                    break
                elif isinstance(value, dict):
                    if any(query in str(v).lower() for v in value.values()):
                        results.append({'id': node_id, **attrs})
                        break
        
        return results 